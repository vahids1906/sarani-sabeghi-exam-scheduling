import csv
import time
import sys
import glob
import os
import pandas as pd
from collections import defaultdict
from itertools import combinations


class Graph:
    def __init__(self):
        self.students = []
        self.courses = set()
        self.graph = defaultdict(list)
        self.degree = defaultdict(int)
        self.course_size = defaultdict(int)
        self.v = 0

    def load_data_from_csv(self, filename):
        """
        CSV format:
        first column  -> student id/name
        remaining columns -> courses taken by the student
        """
        with open(filename, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip header row

            for row in reader:
                if not row:
                    continue

                student = row[0].strip()
                courses = [c.strip() for c in row[1:] if c.strip()]

                if not student or not courses:
                    continue

                # remove duplicate courses for the same student
                courses = list(set(courses))

                self.students.append((student, courses))
                self.courses.update(courses)

        self.courses = sorted(list(self.courses))
        self.v = len(self.courses)

        # Count how many students are enrolled in each course
        for _, courses in self.students:
            for course in courses:
                self.course_size[course] += 1

        # Build conflict graph
        for _, courses in self.students:
            for course1, course2 in combinations(courses, 2):
                if course2 not in self.graph[course1]:
                    self.graph[course1].append(course2)
                if course1 not in self.graph[course2]:
                    self.graph[course2].append(course1)

        # Compute degree
        for course in self.courses:
            self.degree[course] = len(self.graph[course])

    def welsh_powell_coloring(self):
        sorted_vertices = sorted(self.courses, key=lambda x: self.degree[x], reverse=True)

        colors = {}
        color_idx = 1

        for vertex in sorted_vertices:
            if vertex not in colors:
                colors[vertex] = color_idx

                for other in sorted_vertices:
                    if other not in colors:
                        conflict = False
                        for neighbor in self.graph[other]:
                            if neighbor in colors and colors[neighbor] == color_idx:
                                conflict = True
                                break

                        if not conflict:
                            colors[other] = color_idx

                color_idx += 1

        return colors

    def compute_color_student_counts(self, colors):
        color_unique_students = defaultdict(set)

        for student, courses in self.students:
            student_colors = set()

            for c in courses:
                if c in colors:
                    student_colors.add(colors[c])

            for color in student_colors:
                color_unique_students[color].add(student)

        return {color: len(students) for color, students in color_unique_students.items()}

    def compute_color_course_groups(self, colors):
        color_groups = defaultdict(list)

        for course, color in colors.items():
            color_groups[color].append(course)

        for color in color_groups:
            color_groups[color] = sorted(color_groups[color])

        return dict(color_groups)

    def export_schedule_table(self, colors,
                              color_student_counts,
                              excel_file="ETP_Final_Schedule.xlsx",
                              csv_file="ETP_Final_Schedule.csv"):
        color_groups = self.compute_color_course_groups(colors)

        rows = []

        for color in sorted(color_groups):
            courses_in_day = color_groups[color]

            for idx, course in enumerate(courses_in_day):
                row = {
                    "Day/Color": color if idx == 0 else "",
                    "Course": course,
                    "Course_Students": self.course_size[course],
                    "Total_Unique_Students_In_Day": color_student_counts.get(color, 0) if idx == 0 else ""
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        # اول تلاش برای اکسل
        try:
            df.to_excel(excel_file, index=False)
            print(f"\nExcel file saved successfully: {excel_file}")
        except Exception as e:
            print(f"\nExcel export failed: {e}")
            print("Saving CSV instead...")
            df.to_csv(csv_file, index=False, encoding="utf-8-sig")
            print(f"CSV file saved successfully: {csv_file}")

        return df

    def export_text_report(self, colors, color_student_counts, computation_time,
                            conflicts=None, filename="coloring_results.txt",
                            input_filename=""):
        """
        Write the same style of plain-text report the original script produced:
        course->color mapping, colors and their courses, and unique student
        counts per color.
        """
        color_groups = self.compute_color_course_groups(colors)
        conflicts = conflicts or []

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Input file: {input_filename}\n")
            f.write(f"Number of courses (vertices): {self.v}\n")
            f.write(f"Number of colors used: {len(set(colors.values()))}\n")
            f.write(f"Computation time: {computation_time:.4f} seconds\n\n")

            f.write("Course -> Color mapping:\n")
            for course, color in sorted(colors.items()):
                f.write(f"{course}: Color {color}\n")

            f.write("\nColors and their courses:\n")
            for color in sorted(color_groups):
                f.write(f"Color {color}: {color_groups[color]}\n")

            f.write("\nColor -> number of unique students:\n")
            for color in sorted(color_groups):
                count = color_student_counts.get(color, 0)
                f.write(f"Color {color}: {count} students\n")

            if conflicts:
                f.write("\nCONFLICTS DETECTED:\n")
                f.write(f"Number of conflicts: {len(conflicts)}\n")
                for course1, course2, color in conflicts:
                    f.write(f"Conflict: {course1} and {course2} both have color {color}\n")
            else:
                f.write("\nNo conflicts detected! The coloring is valid.\n")

        print(f"Text report saved successfully: {filename}")

    def export_detailed_excel(self, colors, color_student_counts,
                               excel_file="coloring_results.xlsx",
                               csv_prefix="coloring_results"):
        """
        Export the same information as the text report (course->color mapping,
        colors and their courses, color->unique student counts), but as a
        structured Excel workbook with three sheets so it can be loaded
        programmatically (e.g. with pandas.read_excel) in another script.
        """
        color_groups = self.compute_color_course_groups(colors)

        # Sheet 1: one row per course -> its assigned color
        course_color_df = pd.DataFrame(
            sorted(colors.items()),
            columns=["Course", "Color"]
        )

        # Sheet 2: one row per color -> comma-separated list of its courses
        color_courses_df = pd.DataFrame(
            [
                {"Color": color, "Courses": ", ".join(color_groups[color]),
                 "Num_Courses": len(color_groups[color])}
                for color in sorted(color_groups)
            ]
        )

        # Sheet 3: one row per color -> unique student count
        color_students_df = pd.DataFrame(
            [
                {"Color": color, "Unique_Students": color_student_counts.get(color, 0)}
                for color in sorted(color_groups)
            ]
        )

        try:
            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                course_color_df.to_excel(writer, sheet_name="Course_Color", index=False)
                color_courses_df.to_excel(writer, sheet_name="Color_Courses", index=False)
                color_students_df.to_excel(writer, sheet_name="Color_Student_Counts", index=False)
            print(f"Detailed Excel report saved successfully: {excel_file}")
        except Exception as e:
            print(f"\nDetailed Excel export failed: {e}")
            print("Saving as separate CSV files instead...")
            course_color_df.to_csv(f"{csv_prefix}_course_color.csv", index=False, encoding="utf-8-sig")
            color_courses_df.to_csv(f"{csv_prefix}_color_courses.csv", index=False, encoding="utf-8-sig")
            color_students_df.to_csv(f"{csv_prefix}_color_student_counts.csv", index=False, encoding="utf-8-sig")
            print(f"CSV files saved with prefix: {csv_prefix}_*")

        return course_color_df, color_courses_df, color_students_df


def resolve_input_filename():
    """
    Determine the CSV file to load automatically, without prompting the user.

    Priority:
    1. A filename passed as a command-line argument
       (e.g. `python coloring2.py data.csv`)
    2. A file named exactly 'input.csv' in the script's directory
    3. If exactly one .csv file exists in the script's directory, use it
    4. Otherwise, raise an error listing what was found
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Command-line argument
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Specified file not found: {candidate}")

    # 2. Default fixed filename
    default_path = os.path.join(script_dir, 'input.csv')
    if os.path.isfile(default_path):
        return default_path

    # 3. Auto-detect a single CSV file in the script directory
    csv_files = glob.glob(os.path.join(script_dir, '*.csv'))
    if len(csv_files) == 1:
        return csv_files[0]
    elif len(csv_files) > 1:
        names = ", ".join(os.path.basename(f) for f in csv_files)
        raise FileNotFoundError(
            f"Multiple CSV files found ({names}). "
            f"Please specify which one to use: "
            f"python {os.path.basename(__file__)} <filename.csv>"
        )
    else:
        raise FileNotFoundError(
            "No CSV file found. Place a file named 'input.csv' next to the script, "
            f"or run: python {os.path.basename(__file__)} <filename.csv>"
        )


def main():
    try:
        filename = resolve_input_filename()
        print(f"Using input file: {filename}")

        g = Graph()
        g.load_data_from_csv(filename)

        print(f"\nNumber of courses (vertices): {g.v}")

        start_time = time.time()
        colors = g.welsh_powell_coloring()
        computation_time = time.time() - start_time

        color_student_counts = g.compute_color_student_counts(colors)

        print(f"\nComputation time: {computation_time:.4f} seconds")
        print(f"Number of colors used: {len(set(colors.values()))}")

        print("\nColor -> number of unique students:")
        for color in sorted(color_student_counts):
            print(f"Color {color}: {color_student_counts[color]} students")

        # 1) Same schedule table output as before (ETP_Final_Schedule.xlsx)
        g.export_schedule_table(colors, color_student_counts)

        # 2) Plain-text report, same format as coloring_results.txt
        g.export_text_report(colors, color_student_counts, computation_time,
                              input_filename=filename)

        # 3) The text report's info as a structured Excel file for use in other code
        g.export_detailed_excel(colors, color_student_counts)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()