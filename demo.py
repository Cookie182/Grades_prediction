import os
import math
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
from tabulate import tabulate
import pickle
import stdiomask
import questionary

from mysql import connector as mysql

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as gs

# to generate numbers from a given range of numbers with a Gaussian weight distribution
from scipy import stats as ss
# add Savozky-Golay filler data to smoothen out lines on graphs
from scipy.signal import savgol_filter

plt.style.use('bmh')  # matplotlib graph style
sns.set_style('dark')  # seaborn graph style
warnings.filterwarnings('ignore')  # to ignore messages from seaborn graphs


class predictors:
    def grades(test_type, test_amount, max_mark, weightage, pass_percent, final_test_name, n=1000):  # grades generator
        """Func that generates train/test data for the classifier/regressor and returns the trained classifier/regressor"""
        df = pd.DataFrame(index=range(1, n+1))  # making the dataframe and generating dummy marks
        df.index.name = 'Student'
        print("\nGenerating mock data\n")

        passfail_final, overallgrade_final = None, None
        passfail_final_acc, overallgrade_final_acc = 0, 0
        # generating mock data 5 times to find the models with the higehst accuracy
        for test_run in range(1, 6):
            print(f"\nTest Run {test_run}\n")
            for x in range(len(test_type)):
                m = max_mark[x]  # storing max marks for each type of test
                # generating random marks in marking range with a gaussian weight distribution to each mark
                if test_amount[x] > 1:
                    # mew = 65% of max mark and sigma = a third of max mark
                    for y in range(1, test_amount[x] + 1):
                        df[f"{test_type[x]} {y}"] = [round(x) for x in (ss.truncnorm.rvs(((0 - int(m * 0.65)) / (m//3)),
                                                                                         ((m - int(m * 0.65)) / (m//3)),
                                                                                         loc=round(m * 0.65, 0), scale=(m//3), size=n))]
                else:
                    for y in range(1, test_amount[x] + 1):
                        df[f"{test_type[x]}"] = [round(x) for x in (ss.truncnorm.rvs(((0 - int(m * 0.65)) / (m//3)),
                                                                                     ((m - int(m * 0.65)) / (m//3)),
                                                                                     loc=round(m * 0.65, 0), scale=(m//3), size=n))]

            # calculating total grade weight weightage
            df['Total %'] = [0] * len(df)
            for x in range(len(test_type)):
                df['Total %'] += round((df.filter(regex=test_type[x]).sum(axis=1) / (test_amount[x] * max_mark[x])) * weightage[x], 2)

            # determining pass/fail
            df['Pass/Fail'] = ["Pass" if x >= pass_percent else "Fail" for x in df['Total %']]
            print("Generated mock data!\n")

            print(f"\nStudents passed -> {len(df[df['Pass/Fail'] == 'Pass'])}\
            \nStudents Failed -> {len(df[df['Pass/Fail'] == 'Fail'])}\n")

            y = df[['Pass/Fail', 'Total %']].copy()
            y['Pass/Fail'] = LabelEncoder().fit_transform(y['Pass/Fail'])
            X = df[[x for x in df.columns if x not in ['Pass/Fail', 'Total %', final_test_name]]].copy()

            print("Creating and fitting models\n")
            # making train and test data for models
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y['Pass/Fail'])

            # passing probability predictor
            passfail = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.arange(0.1, 1.1, 0.1),
                                                                            cv=RepeatedStratifiedKFold(n_splits=10, random_state=7),
                                                                            max_iter=1000, n_jobs=-1, refit=True, random_state=7,
                                                                            class_weight='balanced')).fit(X_train, y_train['Pass/Fail'])

            # final overall grade predictor
            overallgrade = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1)).fit(X_train, y_train['Total %'])
            print("Models created")

            pf_score = np.round(passfail.score(X_test, y_test['Pass/Fail'])*100, 2)
            og_score = np.round(overallgrade.score(X_test, y_test['Total %'])*100, 2)
            print("LogisticRegressionCV classifer:-")
            print(f"Accuracy -> {pf_score}%")
            print(f"f1_score -> {np.round(f1_score(y_test['Pass/Fail'], passfail.predict(X_test))*100, 2)}%\n")

            print("LinearRegression regressor:-")
            print(f"Accuracy -> {og_score}%")
            print("Models created\n")

            # storing the models that are trained with the best data and the highest test accuracies
            if (pf_score > passfail_final_acc) and (og_score > overallgrade_final_acc):
                passfail_final, overallgrade_final = passfail, overallgrade
            print(' ')

        return passfail_final, overallgrade_final

    # to present rolling predictions based on a student's marks

    def rolling_predict(marks, subject, record):
        # saved to calculate rolling actual grade (calculated with subject structure details)
        all_marks = marks
        # prepping data to be used for predictions
        marks = np.array(marks[:-1]).reshape(1, -1)
        # loading subject prediction models
        passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
        overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

        # making dummy list to cummulatively add each test score
        dummy = [0] * len(marks[0])
        pass_probabs = []  # to store each probability as each test score gets entered
        for x in range(len(marks[0])):
            dummy[x] = marks[0][x]
            pass_probabs.append(passfail.predict_proba(np.array([dummy]))[0][1] * 100)

        # interpolating results to give a smoother graph
        pass_probabs_l = len(pass_probabs)
        if pass_probabs_l % 2 == 0:
            pass_probabs_l -= 1
        pass_probabs = savgol_filter(pass_probabs, pass_probabs_l, 4)

        # limits determiend to scale the pass/fail graph better
        limit1 = math.ceil(max([abs(x - 50) for x in pass_probabs]))

        pf = None  # predicting if in the end the model predicts student failing or passing
        if passfail.predict(marks) == 0:
            pf = 'Fail'
        else:
            pf = 'Pass'

        dummy = [0] * len(marks[0])  # calculating rolling overall mark percentage
        total_percent = []
        for x in range(len(marks[0])):
            dummy[x] = marks[0][x]
            total_percent.append(round(overallgrade.predict(np.array(dummy).reshape(1, -1))[0]*100, 3))

        # interpolating results to give a smoother graph
        total_percent_l = len(total_percent)
        if total_percent_l % 2 == 0:
            total_percent_l -= 1

        total_percent = savgol_filter(total_percent, total_percent_l, 4)

        # limits determined to scale the overall grade graph better
        limit2 = math.ceil(max([abs(x-60) for x in total_percent]))

        # calculating grade
        grade_p = overallgrade.predict(marks)[0]*100
        grade = None
        if grade_p >= 90:
            grade = "A+"
        elif grade_p >= 80:
            grade = "A"
        elif grade_p >= 70:
            grade = "B"
        elif grade_p >= 60:
            grade = "C"
        elif grade_p >= 50:
            grade = "D"
        else:
            grade = "F"

        # calculating the rolling actual grade depending on subject structure
        dummy = [0] * len(all_marks)
        actual_grades = []
        for x in range(len(all_marks)):
            dummy[x] = all_marks[x]
            cursor.execute(f"SELECT Amount FROM {subject}_details")
            amounts = [int(x[0]) for x in cursor.fetchall()]

            cursor.execute(f"SELECT Weightage FROM {subject}_details")
            weightages = [float(x[0]) for x in cursor.fetchall()]

            cursor.execute(f"SELECT Max_mark FROM {subject}_details")
            max_marks = [int(x[0]) for x in cursor.fetchall()]

            # calculating rolling actual grade of student
            mults = []
            c = 0
            for x in range(len(amounts)):
                for y in range(amounts[x]):
                    mults.append((dummy[c] / max_marks[x]) * (weightages[x] / amounts[x]))
                    c += 1
                    del y

            mults = [np.round(x*100, 2) for x in mults]
            p_grade = np.sum(mults)
            actual_grades.append(np.round(p_grade, 2))

        actual_calc_grade = actual_grades[-1]
        # interpolating results to give a smoother graph
        actual_grades_l = len(actual_grades)
        if actual_grades_l % 2 == 0:
            actual_grades_l -= 1

        actual_grades = savgol_filter(actual_grades, actual_grades_l, 4)

        # limits determined to scale the overall grade graph better
        limit3 = math.ceil(max([abs(x-60) for x in actual_grades]))

        actual_grade = None
        if actual_grades[-1] >= 90:
            actual_grade = "A+"
        elif actual_grades[-1] >= 80:
            actual_grade = "A"
        elif actual_grades[-1] >= 70:
            actual_grade = "B"
        elif actual_grades[-1] >= 60:
            actual_grade = "C"
        elif actual_grades[-1] >= 50:
            actual_grade = "D"
        else:
            actual_grade = "F"

        # getting the name of tests used for predictions
        cursor.execute(f"DESCRIBE {record}_{subject}")
        tests = [x[0] for x in cursor.fetchall()][1:-3]

        fig = plt.figure(f"Grade prediction and calculation for {subject}", figsize=(10, 5))
        grid = gs(nrows=2, ncols=2, figure=fig)
        plt.suptitle(f"Chance of Passing and Predicted Total Grade for {subject}\nTake the predictions with a grain of salt", fontsize=12)
        fig.add_subplot(grid[0, 0])
        plt.title(f"Probability of passing the subject after each test taken\nPredicted Pass or Fail? -> {pf}\
        \nChance of passing subject -> {passfail.predict_proba(marks)[0][1] * 100:.2f}%", fontsize=11)
        plt.axhline(50, color='r', label="Threshold", linestyle='--')
        plt.plot(tests[:-1], pass_probabs, c='black',
                 lw=1, label='Predicted passing chance')
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8, rotation=45)
        plt.ylabel('Probability (%)', fontsize=9)
        plt.ylim(ymin=50-limit1, ymax=50+limit1)
        plt.margins(0.02, 0.02)
        plt.legend(loc='best', fontsize=7)
        plt.tight_layout()

        fig.add_subplot(grid[0, 1])
        plt.title(
            f"Predicting Overall grade after each test\nPredicted Overall Subject Grade (out of 100)-> {grade_p:.2f}\nPredicted Grade -> {grade}", fontsize=11)
        plt.axhline(60, color='r', label='Passing Threshold', linestyle='--')
        plt.plot(tests[:-1], total_percent, c='black', lw=1, label='Predicted Overall Grade')
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8, rotation=45)
        plt.ylabel('Final Grade', fontsize=9)
        plt.margins(x=0.01, y=0.01)
        plt.ylim(ymin=60-limit2, ymax=60+limit2)
        plt.legend(loc='best', fontsize=7)
        plt.tight_layout()

        fig.add_subplot(grid[1, :])
        plt.title(f"Actual Rolling Total Mark (out of 100) calculated for {subject} -> {actual_calc_grade} ({actual_grade})", fontsize=11)
        plt.axhline(60, color='r', label='Passing Threshold', linestyle='--')
        plt.plot(tests, actual_grades, c='black', lw=1, label='Caculated Grade (After each test)')
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8, rotation=45)
        plt.ylabel('Grade (out of 100)', fontsize=9)
        plt.margins(x=0.01, y=0.01)
        plt.legend(loc='best', fontsize=7)
        plt.ylim(ymin=60-limit3, ymax=60+limit3)
        plt.tight_layout()
        plt.show()
        print("\nDetailed predictions for marks shown\n")


class student:
    def __init__(self):
        self.record = None
        self.user = None

    def modify_account(self):
        os.system('cls')
        print("Showing account details\n")
        cursor.execute(f"DESCRIBE {self.record}")
        cols = [x[0] for x in cursor.fetchall()]
        for col in cols:
            cursor.execute(f"SELECT {col} FROM {self.record} WHERE username = '{self.user}'")
            print(f"{col} -> {cursor.fetchone()[0]}")

        print('')
        choice = questionary.select("Change account details?: ", choices=['Yes', 'No']).ask()
        if choice == 'Yes':
            cursor.execute(f"DESCRIBE {self.record}")
            cols = [x[0] for x in cursor.fetchall() if x[0] in ["first_name", "last_name", "mobile_no", "username", "password"]]
            cols.append("Go Back")

            print("Which detail to change?\n")
            col = questionary.select("Choices: ", choices=cols).ask()

            if col != "Go Back":
                while True:
                    try:
                        cursor.execute(f"SELECT {col} FROM {self.record} WHERE username = '{self.user}'")
                        old_detail = cursor.fetchall()[0][0]
                        new_detail = input(f"\nOld {col} -> {old_detail}\nEnter new {col} : ")
                        cursor.execute(f"UPDATE {self.record} SET {col} = '{new_detail}' WHERE username = '{self.user}'")
                        break
                    except:
                        print("Enter valid input...\n")

                os.system('cls')
                print(f"Account detail {col} changed from {old_detail} to {new_detail}\n")
            else:
                os.system('cls')
                print("Going back...\n")
        else:
            os.system('cls')
            print(f"Account details for username {self.user} shown\n")

    def view_allsubjectdetails(self):
        os.system('cls')
        while True:
            print("Showing all subjects for student's batch\n")
            cursor.execute(f"SHOW TABLES LIKE '{self.record}_%'")

            # getting subjects from the names of all the marksheets for the student's batch
            subjects = [str(x[0][len(self.record)+1:]).upper()
                        for x in cursor.fetchall()]
            subjects.append("Go Back")

            print("View grades or subject details?\n")
            choice = questionary.select("Choices : ", choices=["View grades",
                                                               "View subject details",
                                                               "Go Back"]).ask()
            if choice == "View grades":
                # getting subject choice
                print("Which subject to view grade for?")
                subject = questionary.select("Choice : ", subjects).ask()
                if subject != "Go Back":
                    # getting column names
                    cursor.execute(f"DESC {self.record}_{subject}")
                    cols = [x[0] for x in cursor.fetchall()]

                    # getting data from marksheet for student
                    cursor.execute(
                        f"SELECT * FROM {self.record}_{subject} WHERE student_id = (SELECT id FROM {self.record} WHERE username = '{self.user}')")
                    marks = cursor.fetchall()[0]

                    print(f"Showing your marks for {subject}\n")
                    [print(f"{cols[x]} -> {marks[x]}")
                        for x in range(len(cols))]
                    input("Press anything to continue...\n")
                    os.system('cls')
                else:
                    os.system('cls')
                    print("Going back...\n")
                    continue

            elif choice == "View subject details":  # viewing subject details taken by the student
                print("Which subject to view details for?")
                subject = questionary.select("Choice : ", subjects).ask()

                if subject != "Go Back":
                    print(tabulate(pd.read_sql(f"SELECT * FROM {subject}_details",
                                               db, index_col='Type'), headers='keys', tablefmt='psql'))
                    print(f"\nDetails shown for {subject}\n")
                    input("Press anything to continue...\n")
                    os.system('cls')
                else:
                    os.system('cls')
                    print("Going back...\n")

            else:
                os.system('cls')
                break

    def custom_predicts(self):
        os.system('cls')
        print("Show predictions with custom grades\n")

        # getting semester choice to display subjects in said semester
        cursor.execute(f"SELECT cur_semester FROM {self.record} WHERE username = '{self.user}'")
        semesters = [str(x) for x in range(1, int(cursor.fetchone()[0])+1)]
        semesters.append("Go Back")

        print("Which semester?\n")
        semester = questionary.select("Choices: ", choices=semesters).ask()
        print(' ')

        if semester != "Go Back":
            # getting subjects that are in the specified semester
            cursor.execute(f"SELECT id, name FROM subjects WHERE semester = '{semester}'")
            subjects = [x[0] for x in cursor.fetchall()]
            subjects.append("Go Back")

            print("Which subject?\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            if subject != "Go Back":
                custom_marks = []  # storing custom marks entered by the student for a subject

                # getting the test types, amount and max_marks for subject
                cursor.execute(f"SELECT type, amount, max_mark FROM {subject}_details")
                subj_details = [x for x in cursor.fetchall()]
                for subj_detail in subj_details:
                    type, amount, max_mark = subj_detail  # unpacking tuple values
                    if amount < 2:  # if it is a test type with one evaluation
                        while True:
                            # making sure the user inputs are valid
                            try:
                                mark = int(input(f"Enter mark for {type} (Max: {max_mark}) -> "))
                                if mark <= max_mark:
                                    custom_marks.append(mark)
                                    break
                                else:
                                    print(f"Mark must be below {max_mark} for {type}!\n")
                            except:
                                print("Enter valid input...\n")
                    else:  # if it is a test type with multiple evaluations
                        for x in range(1, amount+1):
                            while True:
                                try:
                                    mark = int(input(f"Enter mark for {type} {x} (Max: {max_mark}) -> "))
                                    if mark <= max_mark:
                                        custom_marks.append(mark)
                                        break
                                    else:
                                        print(f"Mark must be below {max_mark} for {type}\n")
                                except:
                                    print("Enter valid input...\n")

                # getting predictions and producing graphs with the custom marks
                predictors.rolling_predict(custom_marks, subject, self.record)
                os.system('cls')
                print(f"Custom marks displayed for {self.user} in {subject}\n")

            else:
                os.system('cls')
                print("Going back...\n")

        else:
            os.system('cls')
            print("Going back from viewing custom grades...\n")

    def viewgrades(self):
        os.system('cls')
        print("Viewing your grades\n")

        # getting semester choice to display subjects in said semester
        cursor.execute(f"SELECT cur_semester FROM {self.record} WHERE username = '{self.user}'")
        semesters = [str(x) for x in range(1, int(cursor.fetchall()[0][0])+1)]
        semesters.append("Go Back")

        print("Which semester?\n")
        semester = questionary.select("Choices: ", choices=semesters).ask()
        print(' ')

        if semester != "Go Back":
            # getting subjects that are in the specified semester
            cursor.execute(f"SELECT id, name FROM subjects WHERE semester = '{semester}'")
            subjects = [x[0] for x in cursor.fetchall()]
            subjects.append("Go Back")

            print("Which subject?\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            if subject != "Go Back":
                cursor.execute(f"SELECT id FROM {self.record} WHERE username = '{self.user}'")
                student_id = cursor.fetchone()[0]
                cursor.execute(f"DESCRIBE {self.record}_{subject}")
                tests = [x[0] for x in cursor.fetchall() if x[0] != 'student_id']
                print("")
                for test in tests:  # printing marks on a subject for each test
                    cursor.execute(f"SELECT {test} FROM {self.record}_{subject} WHERE student_id = {student_id}")
                    print(f"{test} -> {cursor.fetchone()[0]}")

                marks = []  # storing marks of student

                choice = questionary.select("Choices: ", choices=["View more info",
                                                                  "Go Back"]).ask()

                if choice == "View more info":  # to show prediction results graphs
                    cursor.execute(f"DESCRIBE {self.record}_{subject}")
                    tests = [x[0] for x in cursor.fetchall()][1:-3]
                    for test in tests:  # iterating and getting marks for tests except for the final  test
                        cursor.execute(f"SELECT {test} FROM {self.record}_{subject} WHERE student_id = {student_id}")
                        marks.append(cursor.fetchone()[0])
                    predictors.rolling_predict(marks, subject, self.record)
                    os.system('cls')
                    print(f"Marks displayed for {self.user} in {subject}\n")
                else:
                    os.system('cls')
                    print(f"Marks displayed for {self.user} in {subject}\n")
            else:
                os.system('cls')
                print("Going back...\n")
        else:
            os.system('cls')
            print("Going back from viewing grades...\n")

    def student_session(self):
        os.system('cls')
        while True:
            cursor.execute(f"SELECT first_name FROM {self.record} WHERE username = '{self.user}'")
            print(f"Welcome to the student menu, {cursor.fetchone()[0]}\n")

            choice = questionary.select("Choices: ", choices=["View grades",
                                                              "Show predictions with custom grades",
                                                              "View all subjects grades/details (including deleted ones)",
                                                              "View account details",
                                                              "Logout"]).ask()

            if choice == "View grades":  # view grades of a subject
                self.viewgrades()

            # taking custom grades for a subject and showing predictions based on them
            elif choice == "Show predictions with custom grades":
                self.custom_predicts()

            # showing marsheets for subjects that may have been deleted
            elif choice == "View all subjects grades/details (including deleted ones)":
                self.view_allsubjectdetails()

            elif choice == "View account details":  # showing/modifying student account details
                self.modify_account()

            elif choice == "Logout":
                os.system('cls')
                print("Logging out from student menu\n")
                break

    def student_auth(self):
        os.system('cls')
        print("Student Login")
        cursor.execute("SELECT start_year FROM students_batch")
        records = [str(x[0]) for x in cursor.fetchall()]
        records.append("Go Back")
        print("\nWhich record do you belong to?\n")
        # choosing which student record they belong to
        record = questionary.select("Choices: ", choices=records).ask()
        if record != "Go Back":
            record = f"students_{record}"
            user = input("Username : ")  # checking login details
            passw = stdiomask.getpass(prompt='Password : ')

            valid_login = False
            # gathering all the student usernames from selected student record
            cursor.execute(f"SELECT username FROM {record}")
            if user in [x[0] for x in cursor.fetchall()]:
                cursor.execute(f"SELECT password FROM {record} WHERE username = '{user}'")
                if passw == cursor.fetchall()[0][0]:
                    valid_login = True
                    self.record = record
                    self.user = user
                    self.student_session()  # launching student session

            if valid_login == False:
                os.system('cls')
                print("Incorrect student login details...\n")


class teacher:
    def __init__(self):
        self.teacher_id = None
        self.teacher_type = None
        self.subjects = None

    def change_subjectdetails(self):
        os.system('cls')
        print("Changing subject details\n")
        subject = questionary.select("Which subject?: ", choices=self.subjects).ask()

        if subject != "Go Back":
            print(f"Are you sure you want to modify {subject}?\n")
            _ = questionary.select("Choices: ", choices=["Yes", "No"]).ask()
            if _ != "No":
                # getting new subject details
                table_name = f"{subject}_details"
                # type of evaluations
                while True:
                    try:
                        print(' ')
                        test_type = []
                        for _ in range(1, int(input("How many types of tests are there? : "))+1):
                            test_type.append(input(f"Enter test type {_} : ").strip().replace(' ', '_'))
                        print(' ')
                        break
                    except:
                        print("Invalid input for test types...\n")

                # amount of tests per each type of evaluation
                while True:
                    try:
                        # amount of tests per evaluation
                        test_amount = tuple(int(input(f"How many {x}?: ")) for x in test_type)
                        print(' ')
                        break
                    except:
                        print("Enter valid inputs for test amounts of each test type...\n")

                # max mark for each type of evaluation
                while True:
                    try:
                        max_mark = tuple(int(input(f"{x} out of how many marks?: ")) for x in test_type)  # maximum marks for each type of tests
                        print(' ')
                        break
                    except:
                        print("Enter valid inputs for max mark of each test type...\n")

                # weightages for each type of evaluation
                while True:
                    weightage = tuple(float(input(f"What is the weightage for {x}?: "))/100 for x in test_type)
                    if np.sum(weightage) == 1.0:
                        print(" ")
                        break
                    else:
                        print("Make sure the weightage for all tests add up to 1.0...\n")

                # passing threshold for subject
                while True:
                    try:
                        pass_percent = float(input("What is the passing percentage threshold?: "))
                        if (pass_percent <= 100) and (pass_percent >= 0):
                            pass_percent = pass_percent/100
                            print(' ')
                            break
                    except:
                        print("Input valid passing percentage threshold...\n")

                # getting the name of the final test evaluation (the final test of the semester for the respective subject)
                final_test_name = test_type[-1]

                # dropping models used for subject with old details
                if os.path.exists(f"{path}/{subject}_passfail"):
                    os.remove(f"{path}/{subject}_passfail")

                if os.path.exists(f"{path}/{subject}_overallgrade"):
                    os.remove(f"{path}/{subject}_overallgrade")

                # dropping table that stored old subject details
                cursor.execute(f"DROP TABLE {table_name}")

                cursor.execute(f"CREATE TABLE {table_name} (Type VARCHAR(30), Amount INT(2), Weightage FLOAT, Max_mark INT(3))")

                passfail, overallgrade = predictors.grades(test_type, test_amount, max_mark, weightage, pass_percent, final_test_name)

                with open(f"{path}/{subject}_passfail", 'wb') as f:
                    pickle.dump(passfail, f)

                with open(f"{path}/{subject}_overallgrade", 'wb') as f:
                    pickle.dump(overallgrade, f)

                # inserting details about new subject
                for x in [tuple((test_type[x], test_amount[x], weightage[x], max_mark[x])) for x in range(len(test_type))]:
                    cursor.execute(f"INSERT INTO {table_name} VALUES ('{x[0]}', {x[1]}, {x[2]}, {x[3]})")

                tests = []  # storing names of tests
                cursor.execute(f"SELECT type, amount FROM {table_name}")
                # getting the column names of tests for subject
                for x in cursor.fetchall():
                    if x[1] > 1:
                        for y in range(1, x[1]+1):
                            tests.append(f"{x[0]}_{y}")
                    else:
                        tests.append(x[0])

                # getting records of students whose mark sheets for this subject should change (whose semesters are on or lower than the semester of subject)
                cursor.execute(
                    f"SELECT start_year FROM students_batch WHERE cur_semester <= (SELECT semester FROM subjects WHERE id = '{subject}')")
                for record in [int(x[0]) for x in cursor.fetchall()]:
                    marksheet = f"students_{record}_{subject}"

                    # storing the old marksheet
                    df = pd.read_sql(f"SELECT * FROM {marksheet}", db, index_col='student_id')

                    # dropping columns other than student_id col on the marksheet (to make room to add new test names)
                    cursor.execute(f"DESC {marksheet}")
                    for old_col in [x[0] for x in cursor.fetchall() if x[0] != 'student_id']:
                        cursor.execute(f"ALTER TABLE {marksheet} DROP COLUMN {old_col}")

                    # adding new tests to marksheet
                    for new_col in tests:
                        cursor.execute(f"ALTER TABLE {marksheet} ADD COLUMN {new_col} INT(3) NOT NULL DEFAULT 0")

                    # adding prediction columns and grade column
                    cursor.execute(f"ALTER TABLE {marksheet} ADD PASS_CHANCE FLOAT NOT NULL DEFAULT 0")

                    cursor.execute(f"ALTER TABLE {marksheet} ADD PREDICTED_GRADE VARCHAR(10) NOT NULL DEFAULT 0")

                    cursor.execute(f"ALTER TABLE {marksheet} ADD GRADE VARCHAR(10) NOT NULL DEFAULT 0")

                    # finding common tests between old and new marksheet
                    common_col = [x for x in df.columns if x in tests]

                    if len(common_col) < 1:
                        os.system('cls')
                        print(f"{subject} details updated\n")
                        continue
                    else:
                        for col in common_col:
                            marks = df[col].values
                            cursor.execute(f"SELECT student_id FROM {marksheet}")
                            ids = [int(x[0]) for x in cursor.fetchall()]
                            for x in range(len(ids)):
                                # updating marks of common tests for respective students between old and new marksheets
                                cursor.execute(f"UPDATE {marksheet} SET {col} = '{marks[x]}' WHERE student_id = '{ids[x]}'")

                        os.system('cls')
                        print(f"Details for {marksheet} added\n")
            else:
                os.system('cls')
                print("Going back...\n")
        else:
            os.system('cls')
            print("Going back...\n")

    def view_subjectdetails(self):
        os.system('cls')
        print("Viewing subject details\n")
        subject = questionary.select("Choices: ", choices=self.subjects).ask()

        if subject != "Go Back":
            table = f"{subject}_details"
            print(tabulate(pd.read_sql(f"SELECT * FROM {table}", db, index_col="Type"), headers='keys', tablefmt='psql'))
            print(f"Details for {subject} shown\n")
            input("\nEnter anything to continue...")
            os.system('cls')
        else:
            os.system('cls')
            print("Invalid option for subject choice, going back\n")

    def manage_account(self):
        os.system('cls')
        print("Changing teacher account details\n")
        df = pd.read_sql(f"SELECT * FROM courses_faculty.teachers WHERE teacher_id = {self.teacher_id}", db, index_col='teacher_id')
        print(tabulate(df, headers='keys', tablefmt='psql'), '\n')
        cursor.execute("DESCRIBE courses_faculty.teachers")
        details = [x[0] for x in cursor.fetchall()]

        for x in ['teacher_id', 'email', 'type', 'username']:  # remove details of account that cannot be changed
            details.remove(x)

        details.append("Go Back")
        print("Which detail to change?")
        detail = questionary.select("Choose: ", choices=details).ask()

        if detail != "Go Back":
            print(f"\nChanging {detail}\n")
            new_detail = input(f"Enter new {detail} : ")
            cursor.execute(
                f"UPDATE courses_faculty.teachers SET {detail} = '{new_detail}' WHERE teacher_id = {self.teacher_id}")

            print("New details")
            df = pd.read_sql(f"SELECT * FROM courses_faculty.teachers WHERE teacher_id = {self.teacher_id}", db, index_col='teacher_id')
            print(tabulate(df, headers='keys', tablefmt='psql'))
            input("Press anything to continue...")

            os.system('cls')
            print(f"Account detail for {detail} updated\n")

        else:
            os.system('cls')
            print("Incorrect choice for account detail to change, going back...\n")

    def view_studentperf(self):
        os.system('cls')
        print("Visually interpreting students' performance for a subject\n")
        cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = '{self.teacher_id}'")

        print("Which subject?\n")
        subject = questionary.select("Choices: ", choices=self.subjects).ask()

        if subject != "Go Back":
            cursor.execute(
                f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id='{subject}')")
            records = [str(x[0]) for x in cursor.fetchall()]
            if len(records) > 0:
                print("Which batch?\n")
                record = questionary.select("Choices: ", choices=records).ask()
                # getting the predicted and actual grades
                df = pd.read_sql(f"SELECT * FROM students_{record}_{subject}", db, index_col='student_id')[['PREDICTED_GRADE', 'GRADE']]

                try:
                    # prediction grades
                    pred_grade_p = np.array([1 if x >= 60 else 0 for x in [float(x.split()[0]) for x in df['PREDICTED_GRADE']]])
                    pred_grade_l = np.array([x.split()[1][1] for x in df['PREDICTED_GRADE']])  # letter grade

                    # actual grades (manually calculated)
                    actual_grade_p = np.array([1 if x >= 60 else 0 for x in [float(x.split()[0]) for x in df['GRADE']]])
                    actual_grade_l = np.array([x.split()[1][1] for x in df['GRADE']])

                    fig = plt.figure()

                    # predictions
                    fig.add_subplot(221)
                    plt.title("Predicted Pass/Fail", fontsize=12)
                    plt.pie([np.count_nonzero(pred_grade_p == 1), np.count_nonzero(pred_grade_p == 0)], labels=['Pass', 'Fail'], autopct='%.2f%%')
                    plt.legend(loc='best')
                    plt.tight_layout()

                    fig.add_subplot(222)
                    plt.title("Predicted Letter Grades", fontsize=12)
                    plt.pie([np.count_nonzero(pred_grade_l == x)
                            for x in np.unique(pred_grade_l)], labels=np.unique(pred_grade_l), autopct='%.2f%%')
                    plt.legend(loc='best')
                    plt.tight_layout()

                    # actual grades
                    fig.add_subplot(223)
                    plt.title("Actual Pass/Fail", fontsize=12)
                    plt.pie([np.count_nonzero(actual_grade_p == 1),
                            np.count_nonzero(actual_grade_p == 0)], labels=['Pass', 'Fail'], autopct='%.2f%%')
                    plt.legend(loc='best')
                    plt.tight_layout()

                    fig.add_subplot(224)
                    plt.title("Actual Letter Grades", fontsize=12)
                    plt.pie([np.count_nonzero(actual_grade_l == x)
                            for x in np.unique(actual_grade_l)], labels=np.unique(actual_grade_l), autopct='%.2f%%')
                    plt.legend(loc='best')
                    plt.tight_layout()
                    plt.show()

                    os.system('cls')
                    print(f"Student's performance shown for {subject}\n")
                except:
                    os.system('cls')
                    print("No predicted grades available, going back...\n")
            else:
                os.system('cls')
                print("There are no student batches who are in the semester (or after) this subject...\n")

        else:
            os.system('cls')
            print("Incorrect subject choice, going back...\n")

    def list_passfail(self):
        os.system('cls')
        print("Listing students who are actually and predicted to pass/fail\n")
        cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = '{self.teacher_id}'")

        print("Which subject?\n")
        subject = questionary.select("Choices: ", choices=self.subjects).ask()

        if subject != "Go Back":
            cursor.execute(
                f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id = '{subject}')")
            records = [str(x[0]) for x in cursor.fetchall()]
            if len(records) > 0:
                print(' ')
                print("Which batch?\n")
                record = questionary.select("Choices: ", choices=records).ask()
                record = f"students_{record}"

                while True:
                    print("\nPredicted or Calculated grades?\n")
                    choice = questionary.select("Choices: ", choices=["Prediction Grades", "Calculated Grades", "Go Back"]).ask()

                    if choice == "Prediction Grades":  # prediction grades
                        os.system('cls')
                        print(f"Showing students in {records[0][1]} who are predicted to pass/fail {subject}\n")
                        print("Predicted to pass:")
                        query = f"SELECT id, first_name, last_name, username, email, entry, PREDICTED_GRADE FROM {record}, {record}_{subject} WHERE {record}.id = {record}_{subject}.student_id AND id IN (SELECT student_id FROM {record}_{subject} WHERE SUBSTRING(PREDICTED_GRADE, 1, 5) >= 60)"

                        df = pd.read_sql(query, db, index_col='id')
                        print(tabulate(df, headers='keys', tablefmt='psql'), '\n')

                        print("Predicted to fail:")
                        query = f"SELECT id, first_name, last_name, username, email, entry, PREDICTED_GRADE FROM {record}, {record}_{subject} WHERE {record}.id = {record}_{subject}.student_id AND id IN (SELECT student_id FROM {record}_{subject} WHERE SUBSTRING(PREDICTED_GRADE, 1, 5) < 60)"

                        df = pd.read_sql(query, db, index_col='id')
                        print(tabulate(df, headers='keys', tablefmt='psql'), '\n')
                        input("Press anything to continue...")
                        os.system('cls')

                    # actual grades (so far)
                    elif choice == "Calculated Grades":
                        os.system('cls')
                        print(f"Showing students in {records[0][1]} and their actual grades (so far) in {subject}\n")
                        print("Going to pass:")
                        query = f"SELECT id, first_name, last_name, username, email, entry, GRADE FROM {record}, {record}_{subject} WHERE {record}.id = {record}_{subject}.student_id AND id IN (SELECT student_id FROM {record}_{subject} WHERE SUBSTRING(GRADE, 1, 5) >= 60)"

                        df = pd.read_sql(query, db, index_col='id')
                        print(tabulate(df, headers='keys', tablefmt='psql'), '\n')

                        print("Going to fail:")
                        query = f"SELECT id, first_name, last_name, username, email, entry, GRADE FROM {record}, {record}_{subject} WHERE {record}.id = {record}_{subject}.student_id AND id IN (SELECT student_id FROM {record}_{subject} WHERE SUBSTRING(GRADE, 1, 5) < 60)"

                        df = pd.read_sql(query, db, index_col='id')
                        print(tabulate(df, headers='keys', tablefmt='psql'), '\n')
                        input("Press anything to continue...")
                        os.system('cls')
                    else:
                        os.system('cls')
                        print("Going back...\n")
                        break
            else:
                os.system('cls')
                print("There are no student batches who are in the semester (or after) this subject...\n")

        else:
            os.system('cls')
            print("Going back...\n")

    def update_studentscore(self):
        os.system('cls')
        print("Update grade for a student\n")
        subject = questionary.select("Choices: ", choices=self.subjects).ask()

        # choosing for which subject to edit marks
        if subject != "Go Back":
            print("")
            cursor.execute(
                f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id = '{subject}')")
            # choosing for which batch to edit marks for
            record = [str(x[0]) for x in cursor.fetchall()]
            if len(record) > 0:
                record.append("Go Back")
                print("Choose student batch\n")
                choice = questionary.select("Choice: ", choices=record).ask()

                if choice != "Go Back":
                    record = f"students_{choice}"
                    print("\nChoose student id to edit marks for : \n")
                    # choosing student
                    cursor.execute(f"SELECT student_id FROM {record}_{subject}")
                    ids = [str(x[0]) for x in cursor.fetchall()]
                    ids.append("Go Back")

                    print("Choose student id\n")
                    student_id = questionary.select("Choices: ", choices=ids).ask()

                    if student_id != "Go Back":
                        cursor.execute(f"SELECT type FROM {subject}_details")
                        test_type = questionary.select("Choices: ", choices=[x[0] for x in cursor.fetchall()]).ask()

                        print(' ')
                        # getting amount for that test type
                        cursor.execute(f"SELECT amount FROM {subject}_details WHERE type = '{test_type}'")
                        amount = int(cursor.fetchall()[0][0])

                        if amount == 1:
                            test = test_type
                        else:
                            print("Which test?\n")
                            test = questionary.select("Choices: ", choices=[f"{test_type}_{x}" for x in range(1, amount+1)]).ask()

                        mark = input(f"\nEnter new mark for student_id {student_id} in {test} for {subject} : ")
                        cursor.execute(f"UPDATE {record}_{subject} SET {test} = {int(mark)} WHERE student_id = {student_id}")

                        # loading subject prediction models
                        passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
                        overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

                        cursor.execute(f"DESCRIBE {record}_{subject}")
                        # get the name of tests for prediction
                        exams = [x[0] for x in cursor.fetchall()][1: -4]
                        mark = []
                        for exam in exams:
                            cursor.execute(f"SELECT {exam} FROM {record}_{subject} WHERE student_id = {student_id}")
                            mark.append(cursor.fetchall()[0][0])

                        # converting to a 2d array so scikit-learn models can use them for predictions
                        mark = np.array(mark).reshape(1, -1)
                        print(mark)
                        # proabability of passing the subject
                        predict_passing = round(passfail.predict_proba(mark)[0][1]*100, 2)
                        # predicted grade of overall subject after finals
                        predict_grade = round(overallgrade.predict(mark)[0]*100, 2)
                        # giving letter based grade based on predicted overall percentage after finals
                        predict_lettergrade = None
                        if predict_grade >= 90:
                            predict_lettergrade = 'A+'
                        elif predict_grade >= 80:
                            predict_lettergrade = 'A'
                        elif predict_grade >= 70:
                            predict_lettergrade = 'B'
                        elif predict_grade >= 60:
                            predict_lettergrade = 'C'
                        elif predict_grade >= 50:
                            predict_lettergrade = 'D'
                        else:
                            predict_lettergrade = 'F'

                        # new predict grade with mark and letter grade
                        predict_grade = f"{predict_grade} ({predict_lettergrade})"

                        cursor.execute(f"UPDATE {record}_{subject} SET PASS_CHANCE = {predict_passing} WHERE student_id = {student_id}")

                        cursor.execute(f"UPDATE {record}_{subject} SET PREDICTED_GRADE = '{predict_grade}' WHERE student_id = {student_id}")

                        # gathering tests to calculate actual grade using subject details and actual current marks
                        cursor.execute(f"SELECT * FROM {record}_{subject} WHERE student_id = {student_id}")
                        marks = [int(x) for x in [x for x in cursor.fetchall()[0]][1:-3]]

                        cursor.execute(f"SELECT Amount FROM {subject}_details")
                        amounts = [int(x[0]) for x in cursor.fetchall()]

                        cursor.execute(f"SELECT Weightage FROM {subject}_details")
                        weightages = [float(x[0]) for x in cursor.fetchall()]

                        cursor.execute(f"SELECT Max_mark FROM {subject}_details")
                        max_marks = [int(x[0]) for x in cursor.fetchall()]

                        # calculating rolling actual grade ef student
                        mults = []
                        c = 0
                        for x in range(len(amounts)):
                            for y in range(amounts[x]):
                                mults.append((marks[c] / max_marks[x]) * (weightages[x] / amounts[x]))
                                c += 1
                        mults = [np.round(x*100, 2) for x in mults]
                        p_grade = np.round(np.sum(mults), 2)
                        l_grade = None
                        if p_grade >= 90:
                            l_grade = 'A+'
                        elif p_grade >= 80:
                            l_grade = 'A'
                        elif p_grade >= 70:
                            l_grade = 'B'
                        elif p_grade >= 60:
                            l_grade = 'C'
                        elif p_grade >= 50:
                            l_grade = 'D'
                        else:
                            l_grade = 'F'

                        grade = f"{p_grade} ({l_grade})"
                        cursor.execute(f"UPDATE {record}_{subject} SET GRADE = '{grade}' WHERE student_id = {student_id}")

                        os.system('cls')
                        print(f"Mark updated for student_id {student_id} in {test} for {subject}\n")

                    else:
                        os.system('cls')
                        print("Going back...\n")
                else:
                    os.system('cls')
                    print("Going back...\n")
            else:
                os.system('cls')
                print("No student batch has reached this subject yet...\n")

        else:
            os.system('cls')
            print("Incorrect choice for subject selection, going back...\n")

    def update_testscore(self):
        os.system('cls')
        print("Updating a test score\n")
        subject = questionary.select("Which subject?: ", choices=self.subjects).ask()

        # choosing for which subject to edit marks
        if subject != "Go Back":
            print("")
            cursor.execute(
                f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id='{subject}')")
            # choosing for which batch to edit marks for
            records = [str(x[0]) for x in cursor.fetchall()]
            if len(records) > 0:
                print("Choose student batch\n")
                records.append("Go Back")
                record = questionary.select("Which batch?: ", choices=records).ask()

                if record != "Go Back":  # choosing student record
                    record = f"students_{record}"
                    cursor.execute(f"SELECT type FROM {subject}_details")
                    print("Which test type?: ")
                    test_type = questionary.select("Choices: ", choices=[x[0] for x in cursor.fetchall()]).ask()

                    print(' ')
                    # getting amount for that test type
                    cursor.execute(f"SELECT amount FROM {subject}_details WHERE type = '{test_type}'")
                    amount = int(cursor.fetchone()[0])

                    # determining name of test, depending on if it is singular type of test or not
                    if amount == 1:
                        test = test_type
                    else:
                        print("Which test?\n")
                        test = questionary.select("Choices: ", choices=[f"{test_type}_{x}" for x in range(1, amount+1)]).ask()

                    # loading subject prediction models
                    passfail = pickle.load(
                        open(f"{path}/{subject}_passfail", 'rb'))
                    overallgrade = pickle.load(
                        open(f"{path}/{subject}_overallgrade", 'rb'))

                    os.system('cls')
                    print(f"Editing marks of {test} for {record} for {subject}\n")
                    # getting the list of student_ids in this mark sheet
                    cursor.execute(f"SELECT Max_Mark FROM {subject}_details WHERE type = '{test_type}'")
                    max_mark = int(cursor.fetchall()[0][0])

                    cursor.execute(f"SELECT student_id FROM {record}_{subject}")
                    # iterating for each student
                    ids = [x[0] for x in cursor.fetchall()]
                    print("Enter marks\n")
                    for student_id in ids:
                        while True:
                            try:
                                mark = int(input(f"Student {student_id} -> "))
                                # update mark of each student
                                if mark > max_mark:
                                    print(f"Enter a valid mark for {test_type}...\n")
                                    continue

                                cursor.execute(f"UPDATE {record}_{subject} SET {test} = {mark} WHERE student_id = {student_id}")

                                break
                            except:
                                print(f"Enter a valid mark for {test_type}...\n")

                    for student_id in ids:
                        cursor.execute(f"DESCRIBE {record}_{subject}")
                        # get the name of tests for prediction
                        tests = [x[0] for x in cursor.fetchall()][1:-4]
                        mark = []
                        for test in tests:
                            cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {student_id}")
                            mark.append(cursor.fetchall()[0][0])

                        # converting to a 2d array so scikit-learn models can use them for predictions
                        mark = np.array(mark).reshape(1, -1)
                        # proabability of passing the subject
                        predict_passing = round(passfail.predict_proba(mark)[0][1]*100, 2)
                        # predicted grade of overall subject after finals
                        predict_grade = round(overallgrade.predict(mark)[0]*100, 2)
                        # giving letter based grade based on predicted overall percentage after finals
                        predict_lettergrade = None
                        if predict_grade >= 90:
                            predict_lettergrade = 'A+'
                        elif predict_grade >= 80:
                            predict_lettergrade = 'A'
                        elif predict_grade >= 70:
                            predict_lettergrade = 'B'
                        elif predict_grade >= 60:
                            predict_lettergrade = 'C'
                        elif predict_grade >= 50:
                            predict_lettergrade = 'D'
                        else:
                            predict_lettergrade = 'F'

                        # new predict grade with mark and letter grade
                        predict_grade = f"{predict_grade} ({predict_lettergrade})"

                        cursor.execute(f"UPDATE {record}_{subject} SET PASS_CHANCE = {predict_passing} WHERE student_id = {student_id}")

                        cursor.execute(f"UPDATE {record}_{subject} SET PREDICTED_GRADE = '{predict_grade}' WHERE student_id = {student_id}")

                        # gathering tests to calculate actual grade using subject details and actual current marks
                        cursor.execute(f"SELECT * FROM {record}_{subject} WHERE student_id = {student_id}")
                        marks = [int(x) for x in [x for x in cursor.fetchall()[0]][1:-3]]

                        cursor.execute(f"SELECT Amount FROM {subject}_details")
                        amounts = [int(x[0]) for x in cursor.fetchall()]

                        cursor.execute(f"SELECT Weightage FROM {subject}_details")
                        weightages = [float(x[0]) for x in cursor.fetchall()]

                        cursor.execute(f"SELECT Max_mark FROM {subject}_details")
                        max_marks = [int(x[0]) for x in cursor.fetchall()]

                        # calculating rolling actual grade of student
                        mults = []
                        c = 0
                        for x in range(len(amounts)):
                            for y in range(amounts[x]):
                                mults.append((marks[c] / max_marks[x]) * (weightages[x] / amounts[x]))
                                c += 1
                        mults = [np.round(x*100, 2) for x in mults]
                        p_grade = np.round(np.sum(mults), 2)
                        l_grade = None
                        if p_grade >= 90:
                            l_grade = 'A+'
                        elif p_grade >= 80:
                            l_grade = 'A'
                        elif p_grade >= 70:
                            l_grade = 'B'
                        elif p_grade >= 60:
                            l_grade = 'C'
                        elif p_grade >= 50:
                            l_grade = 'D'
                        else:
                            l_grade = 'F'

                        grade = f"{p_grade} ({l_grade})"
                        cursor.execute(f"UPDATE {record}_{subject} SET GRADE = '{grade}' WHERE student_id = {student_id}")

                    os.system('cls')
                    print(f"Marks for {test} in {subject} updated\n")

                else:
                    os.system('cls')
                    print("Going back...\n")

            else:
                os.system('cls')
                print("Going back...\n")

    def review_marks(self):
        os.system('cls')
        print("Review overall marks for a subject\n")

        print("Which subject?\n")
        choice = questionary.select("Choices: ", choices=self.subjects).ask()

        # choosing for which subject to edit marks
        if choice != "Go Back":
            print("")
            subject = choice
            cursor.execute(
                f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id = '{subject}')")
            # choosing for which batch to edit marks for
            records = [str(x[0]) for x in cursor.fetchall()]
            records.append("Go Back")

            print("Which batch?\n")
            record = questionary.select("Choices: ", choices=records).ask()
            if len(records) > 0:
                if record != "Go Back":
                    record = f"students_{int(record)}"
                    table = f"{record}_{subject}".lower()
                    print('')
                    print(table)
                    print(tabulate(pd.read_sql(f"SELECT * FROM {table}", db, index_col='student_id'), headers='keys', tablefmt='psql'), '\n')
                    print(f"{record}_{subject} values shown\n")
                    input("Press anything to continue...")
                    os.system('cls')
                else:
                    os.system('cls')
                    print("Going back...\n")

            else:
                os.system('cls')
                print("No student batch have reached this subject yet...\n")

        else:
            os.system('cls')
            print("Going back...\n")

    def teacher_session(self):
        os.system('cls')
        while True:
            if self.teacher_type == "Teacher":
                # if teacher is a teacher
                cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = {self.teacher_id}")
                self.subjects = [x[0] for x in cursor.fetchall()]
                self.subjects.append("Go Back")
                cursor.execute(f"SELECT first_name FROM courses_faculty.teachers WHERE teacher_id = '{self.teacher_id}'")
                first_name = cursor.fetchone()[0]
                print(f"Welcome to the teacher menu, {first_name}\n")
            else:
                # if teacher is a TA
                cursor.execute(f"SELECT id FROM subjects WHERE ta_id = {self.teacher_id}")
                self.subjects = [x[0] for x in cursor.fetchall()]
                self.subjects.append("Go Back")
                cursor.execute(f"SELECT first_name FROM courses_faculty.teachers WHERE teacher_id = '{self.teacher_id}'")
                first_name = cursor.fetchone()[0]
                print(f"Welcome to the teacher menu, {first_name}\n")

            choice = questionary.select("Choices: ", choices=["Review overall marks for a subject",
                                                              "Update a test score",
                                                              "Update a student's score",
                                                              "List Passing/Failing students",
                                                              "View students' performance",
                                                              "Edit account info",
                                                              "View subject details",
                                                              "Change subject details",
                                                              "Logout"]).ask()

            if choice == "Review overall marks for a subject":  # review marks for a subjects
                self.review_marks()

            elif choice == "Update a test score":  # updating scores of students for a test
                self.update_testscore()

            elif choice == "Update a student's score":  # updating grade of one student in a particular subject
                self.update_studentscore()

            # list the students who are actually and predicted to passing/failing
            elif choice == "List Passing/Failing students":
                self.list_passfail()

            elif choice == "View students' performance":  # visually interpreting students' performance in a subject
                self.view_studentperf()

            elif choice == "Edit account info":  # changing first_name, last_name or password of teacher account
                self.manage_account()

            elif choice == "View subject details":  # viewing subject details that the teacher teaches
                self.view_subjectdetails()

            elif choice == "Change subject details":  # changing subject details
                self.change_subjectdetails()

            elif choice == "Logout":  # logging out of teacher session
                os.system('cls')
                print("Logging out of teachers session...\n")
                break

    def teacher_auth(self):
        print("Teacher Login\n")
        user = input("Username : ")
        passw = stdiomask.getpass(prompt='Password : ')

        # checking if both username AND password match respectively
        valid_login = False
        # gathering all the teacher usernames
        cursor.execute("SELECT username FROM courses_faculty.teachers")
        if user in [x[0] for x in cursor.fetchall()]:
            cursor.execute(f"SELECT password FROM courses_faculty.teachers WHERE username = '{user}'")
            if passw == cursor.fetchall()[0][0]:
                cursor.execute(f"SELECT teacher_id FROM courses_faculty.teachers WHERE username = '{user}'")
                self.teacher_id = cursor.fetchall()[0][0]  # storing teacher id

                cursor.execute(f"SELECT type FROM courses_faculty.teachers WHERE teacher_id = '{self.teacher_id}'")
                self.teacher_type = cursor.fetchall()[0][0]  # storing if teacher or TA

                if self.teacher_type == "Teacher":
                    cursor.execute("SELECT teacher_id FROM subjects")
                    if self.teacher_id in [x[0] for x in cursor.fetchall()]:
                        valid_login = True
                        self.teacher_session()

                else:
                    cursor.execute("SELECT ta_id FROM subjects")
                    if self.teacher_id in [x[0] for x in cursor.fetchall()]:
                        valid_login = True
                        self.teacher_session()

        if valid_login == False:
            os.system('cls')
            print("Incorrect teacher login details...\n")


'''-------------------------------------------------------------------------------------------------------------------------------------------'''


class admin:
    def show_tables():
        os.system('cls')
        while True:
            print("Showing and describing details in a table\n")
            cursor.execute("SHOW TABLES")
            tables = [x[0] for x in cursor.fetchall()]
            tables.append("Go Back")

            print("Which table?\n")
            choice = questionary.select("Choices: ", choices=tables).ask()
            if choice == 'Go Back':
                os.system('cls')
                break

            else:
                print("\nDescribing the table\n")
                print(tabulate(pd.read_sql(f"DESC {choice}", db, index_col='Field'), headers='keys', tablefmt='psql'), '\n')

                print("Details within the table\n")
                df = pd.read_sql(f"SELECT * FROM {choice}", db)
                df = df.set_index(df.columns[0])
                print(tabulate(df, headers='keys', tablefmt='psql'), '\n')

                input("Press anything to continue...")
                os.system('cls')
                print(f"{choice} details shown\n")

    def manage_courses():
        os.system('cls')
        while True:
            print("Managing courses\n")

            choice = questionary.select("Choices: ", choices=["Add a course",
                                                              "Go Back"]).ask()
            if choice == "Add a course":
                print("\nAdding a course\n")
                while True:
                    try:
                        course_id = input("Enter course id : ").replace(' ', '_')
                        name = input("Full name of course : ")
                        founded = int(
                            input("What year was the course founded? : "))
                        years = int(
                            input("How many years does the course last? : "))
                        course_type = questionary.select("What type of course is this?", choices=["Bachelors",
                                                                                                  "Masters",
                                                                                                  "PhD"]).ask()

                        cursor.execute(
                            f"INSERT INTO courses_faculty.courses VALUES ('{course_id}', '{name}', {founded}, {years}, '{course_type}')")

                        print(f"{course_id}, {name} successfully added\n")

                        start.prerequisite_tables(course_id)
                        input("Press anything to continue")
                        os.system('cls')
                        break
                    except:
                        print("Enter appropriate values for each field...\n")

            else:
                os.system('cls')
                print("Going back...\n")
                break

    def add_subject():
        print("\nAdding a subject\n")
        cursor.execute("SELECT id FROM subjects")
        subject_names = [x[0] for x in cursor.fetchall()]
        subj_name = input("Enter abbreviation of subject : ").strip().upper()

        if subj_name not in subject_names:
            try:
                full_name = input("Enter full name of subject : ").rstrip()
                semester = int(input("Which semester is this subject in : "))
                cursor.execute(f"INSERT INTO subjects (id, name, semester) VALUES ('{subj_name}', '{full_name}', {semester})")

                table_name = f"{subj_name}_details"
                # type of evaluations
                while True:
                    try:
                        print(' ')
                        test_type = []
                        for _ in range(1, int(input("How many types of tests are there? : "))+1):
                            test_type.append(input(f"Enter test type {_} : ").strip().replace(' ', '_'))
                        print(' ')
                        break
                    except:
                        print("Invalid input for test types...\n")

                # amount of tests per each type of evaluation
                while True:
                    try:
                        # amount of tests per evaluation
                        test_amount = tuple(int(input(f"How many {x} evaluations?: ")) for x in test_type)
                        print(' ')
                        break
                    except:
                        print("Enter valid inputs for test amounts of each test type...\n")

                # max mark for each type of evaluation
                while True:
                    try:
                        # maximum marks for each type of tests
                        max_mark = tuple(int(input(f"{x} out of how many marks?: ")) for x in test_type)
                        print(' ')
                        break
                    except:
                        print("Enter valid inputs for max mark of each test type...\n")

                # weightages for each type of evaluation
                while True:
                    weightage = tuple(float(input(f"What is the weightage for {x}?: "))/100 for x in test_type)
                    if np.sum(weightage) == 1.0:
                        print(" ")
                        break
                    else:
                        print("Make sure the weightage for all tests add up to 1.0...\n")

                # passing threshold for subject
                while True:
                    try:
                        pass_percent = float(input("What is the passing percentage threshold?: "))
                        if (pass_percent <= 100) and (pass_percent >= 0):
                            pass_percent = pass_percent/100
                            print(' ')
                            break
                    except:
                        print("Input valid passing percentage threshold...\n")

                # getting the name of the final test evaluation (the final test of the semester for the respective subject)
                final_test_name = test_type[-1]

                cursor.execute(f"CREATE TABLE {table_name} (Type VARCHAR(30), Amount INT(2), Weightage FLOAT, Max_mark INT(3))")

                passfail, overallgrade = predictors.grades(test_type, test_amount, max_mark, weightage, pass_percent, final_test_name)

                with open(f"{path}/{subj_name}_passfail", 'wb') as f:
                    pickle.dump(passfail, f)

                with open(f"{path}/{subj_name}_overallgrade", 'wb') as f:
                    pickle.dump(overallgrade, f)

                # inserting details about new subject
                for x in [tuple((test_type[x], test_amount[x], weightage[x], max_mark[x])) for x in range(len(test_type))]:
                    cursor.execute(f"INSERT INTO {table_name} VALUES ('{x[0]}', {x[1]}, {x[2]}, {x[3]})")

                print(f"Details for {full_name} added\n")

                # getting appropriate student record
                cursor.execute(f"SELECT start_year FROM students_batch WHERE cur_semester <= {semester}")
                tables = [x[0] for x in cursor.fetchall()]
                if len(tables) > 0:
                    # making marking sheets for subjects for all students who have a student record
                    for table in tables:
                        table = f"students_{table}"
                        new_table = f"{table}_{subj_name}"

                        cursor.execute(f"SHOW TABLES LIKE '{new_table}'")
                        # excluding student records who already have the marksheet for this subject
                        if len([x[0] for x in cursor.fetchall()]) < 1:
                            tests = []
                            cursor.execute(f"SELECT type, amount FROM {table_name}")
                            # getting the column names of tests for subject
                            for x in cursor.fetchall():
                                if x[1] > 1:
                                    for y in range(1, x[1]+1):
                                        tests.append(f"{x[0]}_{y}")
                                else:
                                    tests.append(x[0])

                            # creating new mark sheet for students doing that particular subject
                            cursor.execute(f"CREATE TABLE IF NOT EXISTS {new_table} (student_id INT(5) PRIMARY KEY)")

                            # adding foreign key to link student ids together
                            cursor.execute(
                                f"ALTER TABLE {new_table} ADD FOREIGN KEY (student_id) REFERENCES {table}(id) ON DELETE CASCADE ON UPDATE CASCADE")

                            # adding columns of tests and making the default marks 0
                            for test in tests:
                                cursor.execute(f"ALTER TABLE {new_table} ADD {test} INT(3) NOT NULL DEFAULT 0")

                            # custom columns to store predictor variables
                            cursor.execute(f"ALTER TABLE {new_table} ADD PASS_CHANCE FLOAT NOT NULL DEFAULT 0")

                            cursor.execute(f"ALTER TABLE {new_table} ADD PREDICTED_GRADE VARCHAR(10) NOT NULL DEFAULT 0")

                            cursor.execute(f"ALTER TABLE {new_table} ADD GRADE VARCHAR(10) NOT NULL DEFAULT 0")

                            # adding each student id for each student record table and this subject
                            cursor.execute(f"SELECT id FROM {table}")
                            for x in [x[0] for x in cursor.fetchall()]:
                                cursor.execute(f"INSERT INTO {new_table} (student_id) VALUES ({x})")

                            cursor.execute(
                                f"CREATE TRIGGER {new_table} AFTER INSERT ON {table} FOR EACH ROW INSERT INTO {new_table} (student_id) values (new.id)")

                    print(f"Grades sheets for {subj_name} created, but no teacher is assigned to subject yet\n")
                    input("Press anything to continue...")
                    os.system('cls')
                else:
                    os.system('cls')
                    print(f"{subj_name} created but no student record tables found...\n")
            except:
                cursor.execute(f"DELETE FROM subjects WHERE id = '{subj_name.upper()}'")

                # selects appropriate batches to add ths subjects for
                cursor.execute("SELECT start_year FROM students_batch")
                for year in [int(x[0]) for x in cursor.fetchall()]:
                    cursor.execute(f"DROP TRIGGER IF EXISTS students_{year}_{subj_name.upper()}")

                cursor.execute(f"SHOW TABLES LIKE '%{subj_name}%'")
                for table in [x[0] for x in cursor.fetchall()]:
                    cursor.execute(f"DROP TABLE {table}")

                os.system('cls')
                print("Enter valid subject details...\n")
        else:
            os.system('cls')
            print(f"{subj_name} already exists in the subjects table...\n")

    def delete_subject():
        os.system('cls')
        print("Deleting a subject\n")
        cursor.execute("SELECT id FROM subjects")
        subjects = [x[0] for x in cursor.fetchall()]
        subjects.append("Go Back")

        print("Which subject to delete?\n")
        choice = questionary.select("Choices: ", choices=subjects).ask()

        if choice != "Go Back":
            subject = choice
            print(f"\nAre you sure you want to delete {subject}?\n")
            choice = questionary.select("Choices: ", choices=["Yes", "No"]).ask()

            if choice == "Yes":
                cursor.execute(f"DELETE FROM subjects WHERE id = '{subject}'")

                # selecting appropriate student batches
                cursor.execute(
                    f"SELECT start_year FROM students_batch WHERE cur_semester <= (SELECT semester FROM subjects where id = '{subject}')")
                records = [f"students_{x[0]}" for x in cursor.fetchall()]
                for record in records:
                    cursor.execute(f"DROP TABLE IF EXISTS {record}_{subject}")

                    cursor.execute(f"SHOW TABLES LIKE '%{subject}%'")
                    tables = [x[0] for x in cursor.fetchall()]
                    if len(tables) > 0:
                        for table in tables:
                            cursor.execute(f"DROP TABLE {table}")

                    cursor.execute(f"DROP TRIGGER IF EXISTS {record}_{subject}")

                if os.path.exists(f"{path}/{subject}_passfail"):
                    os.remove(f"{path}/{subject}_passfail")

                if os.path.exists(f"{path}/{subject}_overallgrade"):
                    os.remove(f"{path}/{subject}_overallgrade")

                os.system('cls')
                print(f"{subject} has been deleted\n")

            else:
                os.system('cls')
                print("Going back...\n")

        else:
            os.system('cls')
            print("Going back...\n")

    def add_teacher():
        os.system('cls')
        print("Adding a teacher\n")
        first = input("First name : ")
        last = input("Last name : ")
        user = input("Enter username : ")
        ta = questionary.select("Is this teacher a TA?", choices=["Yes", "No"]).ask()

        cursor.execute("SELECT username FROM courses_faculty.teachers")
        existing_users = [x[0] for x in cursor.fetchall()]
        if user not in existing_users:
            passw = input("Enter password : ")
            email = f"{first}.{last}@dypiu.ac.in"

            # adding teacher as a normal teacher or TA
            if ta == "No":
                cursor.execute(
                    f"INSERT INTO courses_faculty.teachers(first_name, last_name, username, email, password) VALUES('{first}', '{last}', '{user}', '{email}', '{passw}')")

                os.system('cls')
                print(f"{user} has been added as a teacher\n")

            else:
                cursor.execute(
                    f"INSERT INTO courses_faculty.teachers(first_name, last_name, username, email, password, type) VALUES('{first}', '{last}', '{user}', '{email}', '{passw}', 'Teaching Assistant')")

                os.system('cls')
                print(f"{user} has been added as a teaching assistant\n")

        else:
            os.system('cls')
            print(f"{user} already exists, going back...\n")

    def assign_teacher():
        os.system('cls')
        print("Add teacher to subject\n")
        cursor.execute("SELECT id FROM subjects")
        subj_choices = [x[0] for x in cursor.fetchall()]
        subj_choices.append("Go Back")
        subj = questionary.select("Which subject?: ", choices=subj_choices).ask()
        print(tabulate(pd.read_sql(f"SELECT * FROM subjects WHERE id = '{subj}'", db, index_col='id'), headers='keys', tablefmt='psql'), '\n')

        # getting teacher id and TA id for subjects
        cursor.execute(f"SELECT teacher_id FROM subjects WHERE id = '{subj}'")
        teacher_id = cursor.fetchall()[0][0]

        cursor.execute(f"SELECT ta_id FROM subjects WHERE id = '{subj}'")
        ta_id = cursor.fetchall()[0][0]
        if (ta_id is None) or (teacher_id is None):
            choice = questionary.select("Assign?", choices=["Yes", "No"]).ask()
            if choice == "Yes":
                choice = questionary.select("Assign teacher?", choices=['Yes', 'No']).ask()

                # getting details of teachers who are teachers and not TA's to assign to subjects with no teachers assigned
                if choice == "Yes":
                    print('\n', tabulate(pd.read_sql(f"SELECT teacher_id, first_name, last_name, email FROM courses_faculty.teachers WHERE type = 'Teacher'",
                          db, index_col='teacher_id'), headers='keys', tablefmt='pqsl'))

                    cursor.execute(f"SELECT teacher_id FROM courses_faculty.teachers WHERE type = 'Teacher'")
                    ids = [str(x[0]) for x in cursor.fetchall()]
                    id = questionary.select("Which teacher?", ids).ask()
                    cursor.execute(f"UPDATE subjects SET teacher_id = '{id}' WHERE id = '{subj}'")

                    print(f"Teacher id {id} added as teacher to {subj}\n")

                choice = questionary.select("Assign TA?", choices=['Yes', 'No']).ask()

                # getting details of TAs and assigning them to subjects with no TAs assigned
                if choice == "Yes":
                    print('\n', tabulate(pd.read_sql(f"SELECT teacher_id, first_name, last_name, email FROM courses_faculty.teachers WHERE type != 'Teacher'",
                          db, index_col='teacher_id'), headers='keys', tablefmt='pqsl'))

                    cursor.execute(f"SELECT teacher_id FROM courses_faculty.teachers WHERE type != 'Teacher'")
                    ids = [str(x[0]) for x in cursor.fetchall()]
                    id = questionary.select("Which teacher?", ids).ask()
                    cursor.execute(f"UPDATE subjects SET ta_id = '{id}' WHERE id = '{subj}'")

                    print(f"Teacher id {id} added as TA to {subj}\n")

                os.system('cls')
                print(f"Faculty for {subj} modified...\n")
            else:
                os.system('cls')
                print("Going back...\n")
        else:
            os.system('cls')
            print(f"Teachers and TA already assigned for {subj}, going back...\n")

    def delete_teacher():
        os.system('cls')
        print("Deleting a teacher\n")
        print(tabulate(pd.read_sql("SELECT * FROM courses_faculty.teachers", db,
                                   index_col='teacher_id'), headers='keys', tablefmt='psql'), '\n')
        cursor.execute("SELECT teacher_id FROM courses_faculty.teachers")
        print("Choose id of teacher to delete\n")
        id = int(questionary.select("Choices: ", choices=[str(x[0]) for x in cursor.fetchall()]).ask())

        print("Are you sure?\n")
        choice = questionary.select("Choices: ", choices=["Yes", "No"]).ask()
        if choice == "Yes":
            cursor.execute(f"SELECT username FROM teachers WHERE courses_faculty.teacher_id = '{id}'")
            user = cursor.fetchall()[0][0]
            cursor.execute(f"DELETE FROM teachers WHERE courses_faculty.teacher_id = '{id}'")

            os.system('cls')
            print(f"{user} has been deleted from teacher records\n")
        else:
            os.system('cls')
            print("Invalid choice, going back...\n")

    def unassign_teacher():
        os.system('cls')
        print("Unassign teacher from subject\n")
        cursor.execute("SELECT id FROM subjects")
        subj_choices = [x[0] for x in cursor.fetchall()]
        subj_choices.append("Go Back")
        subj = questionary.select("Which subject?: ", choices=subj_choices).ask()

        # getting teacher id and TA id for subjects
        cursor.execute(f"SELECT teacher_id FROM subjects WHERE id = '{subj}'")
        teacher_id = cursor.fetchall()[0][0]

        cursor.execute(f"SELECT ta_id FROM subjects WHERE id = '{subj}'")
        ta_id = cursor.fetchall()[0][0]

        # can only unassign for subjects with no teachers assigned
        if (teacher_id is not None) or (ta_id is not None):
            # if there is a teacher assigned
            if teacher_id is not None:
                print(tabulate(pd.read_sql(
                    f"SELECT * FROM subjects WHERE id = '{subj}'", db, index_col='id'), headers='keys', tablefmt='psql'), '\n')

                choice = questionary.select("Unassign teacher from subject?", choices=['Yes', 'No']).ask()
                if choice == 'Yes':
                    choice = questionary.select("Are you sure?", choices=['Yes', 'No']).ask()
                    if choice == 'Yes':
                        cursor.execute(f"UPDATE subjects SET teacher_id = NULL WHERE id = '{subj}'")

                        os.system('cls')
                        print(f"Teacher unassigned from {subj}\n")
                    else:
                        os.system('cls')
                        print("Going back...\n")

                else:
                    os.system('cls')
                    print("Going back...\n")

            # if teachign assistant is assigned
            if ta_id is not None:
                print(tabulate(pd.read_sql(
                    f"SELECT * FROM subjects WHERE id = '{subj}'", db, index_col='id'), headers='keys', tablefmt='psql'), '\n')

                choice = questionary.select("Unassign teaching assistant from subject?", choices=['Yes', 'No']).ask()
                if choice == 'Yes':
                    choice = questionary.select("Are you sure?", choices=['Yes', 'No']).ask()
                    if choice == 'Yes':
                        cursor.execute(f"UPDATE subjects SET ta_id = NULL WHERE id = '{subj}'")

                        os.system('cls')
                        print(f"Teaching assistant unassigned from {subj}\n")
                    else:
                        os.system('cls')
                        print("Going back...\n")

                else:
                    os.system('cls')
                    print("Going back...\n")

        else:
            os.system('cls')
            print("No teachers assigned to the subject...\n")

    def new_studentbatch():
        os.system('cls')
        print("Creating new student record\n")
        while True:  # to get valid year input from user
            try:
                year = int(input("Enter year to create student record for : "))
                break
            except:
                os.system('cls')
                print("Enter valid input for year...\n")
        cursor.execute("SELECT start_year FROM students_batch")

        # checking if batch year already exists
        if year not in [x[0] for x in cursor.fetchall()]:
            cursor.execute("SELECT DATABASE()")  # getting name of current course
            cursor.execute(f"SELECT years FROM courses_faculty.courses WHERE id = '{cursor.fetchone()[0]}'")  # getting length of course
            course_len = int(cursor.fetchone()[0])

            # creating entry for students batch table
            cursor.execute(f"INSERT INTO students_batch (start_year, grad_year) VALUES ({int(year)}, {int(year) + course_len})")

            # name of new student records
            table = f"students_{year}"
            # creating student record for that batch
            cursor.execute(
                f"CREATE TABLE {table}(id INT(3) NOT NULL AUTO_INCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, mobile_no VARCHAR(15) DEFAULT NULL, email VARCHAR(40) DEFAULT NULL, username VARCHAR(20) UNIQUE NOT NULL, password VARCHAR(20) NOT NULL, start_year INT NOT NULL DEFAULT {int(year)},start_sem INT NOT NULL DEFAULT 1, cur_semester INT NOT NULL DEFAULT 1, entry VARCHAR(20) DEFAULT 'Normal', grad_year INT DEFAULT {int(year)+4}, PRIMARY KEY(id), UNIQUE(username))")

            # foreign key referencing students_batch table to student record
            cursor.execute(f"ALTER TABLE {table} ADD FOREIGN KEY (start_year) REFERENCES students_batch(start_year)")

            # triggers for counting normal entry students after either deleting/inserting new data for each record
            cursor.execute(
                f"CREATE TRIGGER normal_student_count_{year}_insert AFTER INSERT ON {table} FOR EACH ROW UPDATE students_batch SET students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Normal') WHERE start_year={int(year)}")

            cursor.execute(
                f"CREATE TRIGGER normal_student_count_{year}_delete AFTER DELETE ON {table} FOR EACH ROW UPDATE students_batch SET students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Normal') WHERE start_year={int(year)}")

            # triggers for counting lateral entry students after either deleting/inserting new data for each record
            cursor.execute(
                f"CREATE TRIGGER lateral_student_count_{year}_insert AFTER INSERT ON {table} FOR EACH ROW UPDATE students_batch SET lat_students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Lateral') WHERE start_year={int(year)}")

            cursor.execute(
                f"CREATE TRIGGER lateral_student_count_{year}_delete AFTER DELETE ON {table} FOR EACH ROW UPDATE students_batch SET lat_students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Lateral') WHERE start_year={int(year)}")

            # trigger to update semester count between student records table and students_batch table
            cursor.execute(
                f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {int(year)})")

            # create 2 events to auto increment the semester count for each student batch
            cursor.execute(
                f"CREATE EVENT IF NOT EXISTS sem_count_{year}_jan ON SCHEDULE EVERY 1 YEAR STARTS '{int(year)+1}-01-01' ENDS '{int(year)+course_len}-01-01' DO UPDATE students_batch SET cur_semester = cur_semester + 1 WHERE start_year = {year}")

            cursor.execute(
                f"CREATE EVENT IF NOT EXISTS sem_count_{year}_aug ON SCHEDULE EVERY 1 YEAR STARTS '{int(year)+1}-08-01' ENDS '{int(year)+(course_len-1)}-08=01' DO UPDATE students_batch SET cur_semester = cur_semester + 1 WHERE start_year = {year}")

            cursor.execute("SELECT id FROM subjects")
            subjects = [x[0] for x in cursor.fetchall()]
            for subject in subjects:
                # name of subjects to iterate for making grade sheets
                table_name = f"{subject}_details"
                # name of table to store grades on subject for new students
                new_table = f"{table}_{subject}"

                tests = []
                cursor.execute(f"SELECT type, amount FROM {table_name}")
                # getting the column names of tests for subject
                for x in cursor.fetchall():
                    if x[1] > 1:
                        for y in range(1, x[1]+1):
                            tests.append(f"{x[0]}_{y}")
                    else:
                        tests.append(x[0])

                # creating new mark sheet for students doing that particular subject
                cursor.execute(f"CREATE TABLE {new_table} (student_id INT(5) PRIMARY KEY)")

                # adding foreign key to link student ids together
                cursor.execute(
                    f"ALTER TABLE {new_table} ADD FOREIGN KEY (student_id) REFERENCES {table}(id) ON DELETE CASCADE ON UPDATE CASCADE")

                # adding columns of tests and making the default marks 0
                for test in tests:
                    cursor.execute(f"ALTER TABLE {new_table} ADD {test} INT(3) NOT NULL DEFAULT 0")

                # custom columns to store predictor variables
                cursor.execute(f"ALTER TABLE {new_table} ADD PASS_CHANCE FLOAT NOT NULL DEFAULT 0")

                cursor.execute(f"ALTER TABLE {new_table} ADD PREDICTED_GRADE VARCHAR(10) NOT NULL DEFAULT 0")

                cursor.execute(f"ALTER TABLE {new_table} ADD GRADE VARCHAR(10) NOT NULL DEFAULT 0")

                # adding each student id for each student record table and this subject
                cursor.execute(f"SELECT id FROM {table}")
                for x in [x[0] for x in cursor.fetchall()]:
                    cursor.execute(f"INSERT INTO {new_table} (student_id) VALUES ({x})")

                cursor.execute(
                    f"CREATE TRIGGER {new_table} AFTER INSERT ON {table} FOR EACH ROW INSERT INTO {new_table} (student_id) values (new.id)")

            os.system('cls')
            print(f"{table} records created\n")
        else:
            os.system('cls')
            print(f"Student record for {year} already exists...\n")

    def add_student():
        os.system('cls')
        print("Adding a student to a record\n")
        cursor.execute("SELECT start_year FROM students_batch")
        choices = [str(x[0]) for x in cursor.fetchall()]
        choices.append("Go Back")

        print("Which batch?\n")
        choice = questionary.select("Choice: ", choices=choices).ask()

        if choice != "Go Back":
            student_record = f"students_{int(choice)}"
            cursor.execute(f"SELECT username FROM {student_record}")
            existing_users = [x[0] for x in cursor.fetchall()]
            year = int(choice)
            while True:
                user = input("\nEnter username for new student : ")
                if user not in existing_users:
                    email = f"{user}_{year}@dypiu.ac.in"
                    while True:
                        try:
                            sem = int(input("Which semester did the student join from? : "))
                        except:
                            os.system('cls')
                            print("Enter valid value for semester\n")
                            continue

                        if sem == 1:
                            cursor.execute(f"DROP TRIGGER IF EXISTS semester_count_{year}")

                            cursor.execute(
                                f"INSERT INTO {student_record} (first_name, last_name, email, username, password, start_sem, entry) VALUES ('{input('Enter first name: ')}', '{input('Enter last name: ')}', '{email}', '{user}', '{input('Enter password: ')}', {sem}, 'Normal')")

                            cursor.execute(
                                f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE        students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {year})")
                            break
                        else:
                            cursor.execute(f"DROP TRIGGER IF EXISTS semester_count_{year}")

                            cursor.execute(
                                f"INSERT INTO {student_record} (first_name, last_name, email, username, password, start_sem, entry) VALUES ('{input('Enter first name: ')}', '{input('Enter last name: ')}', '{email}', '{user}', '{input('Enter password: ')}', {sem}, 'Lateral')")

                            cursor.execute(
                                f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE        students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {year})")
                            break

                    cursor.execute(f"SELECT students + lat_students FROM students_batch WHERE start_year = {int(year)}")
                    tot_students = cursor.fetchall()[0][0]
                    cursor.execute(f"UPDATE students_batch SET tot_students = {tot_students} WHERE start_year = {int(year)}")

                    os.system('cls')
                    print(f"{user} has been added as a student for the {year} batch\n")
                    break
                else:
                    os.system('cls')
                    print(f"{user} already exists, try again\n")
        else:
            os.system('cls')
            print("Going back...\n")

    def delete_student():
        os.system('cls')
        print("Deleting a student from a record\n")
        cursor.execute("SELECT start_year FROM students_batch")
        choices = [str(x[0]) for x in cursor.fetchall()]
        choices.append("Go Back")

        print("Which batch?\n")
        choice = questionary.select("Choices: ", choices=choices).ask()

        if choice != "Go Back":
            year = int(choice)
            student_record = f"students_{year}"
            df = pd.read_sql(
                f"SELECT id, first_name, last_name, mobile_no, email, username, start_year, start_sem, cur_semester, entry, grad_year FROM {student_record}", db, index_col='id')
            print(tabulate(df, headers='keys', tablefmt='psql'), '\n')
            user = input("Enter username of student to delete : ")

            cursor.execute(f"SELECT username FROM {student_record}")
            users = [x[0] for x in cursor.fetchall()]
            if user in users:
                os.system('cls')
                print(f"\nAre you sure to delete {user}?")
                print(tabulate(df[df['username'] == user], headers='keys', tablefmt='psql'), '\n')

                choice = questionary.select("Choices: ", choices=["Yes", "No"]).ask()
                if choice == "Yes":
                    cursor.execute(f"DROP TRIGGER IF EXISTS semester_count_{year}")

                    cursor.execute(f"DELETE FROM {student_record} WHERE username = '{user}'")

                    cursor.execute(
                        f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {year})")

                    cursor.execute(f"SELECT students + lat_students FROM students_batch WHERE start_year = {year}")
                    tot_students = cursor.fetchall()[0][0]
                    cursor.execute(f"UPDATE students_batch SET tot_students = {tot_students} WHERE start_year = {year}")

                    os.system('cls')
                    print(f"{user} has been removed from {student_record}\n")
                else:
                    os.system('cls')
                    print("Going back...\n")
            else:
                print(f"Invalid user: {user}, going back...\n")
        else:
            os.system('cls')
            print("Going back...\n")

    def manage_account(user):
        print("Viewing account details\n")
        print(tabulate(pd.read_sql(
            f"SELECT * FROM courses_faculty.admins WHERE username = '{user}'", db, index_col='id'), headers='keys', tablefmt='psql'), '\n')

        # asking whether to change account details or not
        choice = questionary.select("Choice : ", choices=["Change account details", "Go Back"]).ask()

        if choice == "Change account details":
            cursor.execute("DESC courses_faculty.admins")

            # getting account details that can be changed
            cols = [x[0] for x in cursor.fetchall() if x[0] not in ['id', 'email', 'added_by', 'username']]
            cols.append("Go Back")

            print("Which detail to change?\n")
            col = questionary.select("Choice : ", cols).ask()

            if col != "Go Back":
                while True:
                    try:
                        cursor.execute(f"SELECT id FROM courses_faculty.admins WHERE username = '{user}'")
                        id = cursor.fetchall()[0][0]

                        cursor.execute(f"SELECT {col} FROM courses_faculty.admins WHERE username = '{user}'")
                        old_detail = cursor.fetchall()[0][0]
                        new_detail = input(f"\nOld {col} -> {old_detail}\nNew {col} -> ")

                        cursor.execute(
                            f"UPDATE courses_faculty.admins SET courses_faculty.admins.{col} = '{new_detail}' WHERE courses_faculty.admins.username = '{user}'")

                        cursor.execute(f"SELECT username FROM courses_faculty.admins WHERE id = '{id}'")
                        user = cursor.fetchall()[0][0]
                        os.system('cls')
                        print(f"{col} changed for {user}\n")
                        break
                    except:
                        print("Enter valid input...\n")

            else:
                os.system('cls')
                print("Going back...\n")

        elif choice == "Go Back":
            os.system('cls')
            print("Going back...\n")

    def add_admin(user):
        print("Adding an admin account\n")
        os.system('cls')

        while True:
            try:
                first_name = input("Enter first name : ")
                last_name = input("Enter last name : ")
                username = input("Enter username : ")
                passw = input("Enter password : ")
                email = f"{username}_admin@dypiu.ac.in"

                cursor.execute(
                    f"INSERT INTO courses_faculty.admins (first_name, last_name, username, password, email, added_by) VALUES ('{first_name}', '{last_name}', '{username}', '{passw}', '{email}', '{user}')")

                print(f"{username} has been added as an admin\n")
                os.system('cls')
                break
            except:
                print("Enter valid details...\n")

    def admin_session(user):
        while True:
            print(f"Welcome to the admin menu, {user}\n")
            choice = questionary.select("Choices: ", choices=["Show tables",
                                                              "Manage subjects",
                                                              "Manage courses",
                                                              "Manage teacher accounts",
                                                              "Manage student accounts",
                                                              "Manage my account details",
                                                              "Add admin account",
                                                              "Logout"]).ask()

            if choice == 'Show tables':  # showing all table names and details about them
                admin.show_tables()

            elif choice == "Manage courses":  # adding or modifying courses
                admin.manage_courses()

            elif choice == 'Manage subjects':  # adding or deleting a subject
                os.system('cls')
                print("Managing subjects\n")
                choice = questionary.select("Choices: ", choices=["Add a subject",
                                                                  "Delete a subject",
                                                                  "Go Back"]).ask()

                # adding a subject, however, not setting a teacher/TA for it yet
                if choice == 'Add a subject':
                    admin.add_subject()

                elif choice == 'Delete a subject':  # deleting a subject
                    admin.delete_subject()

                elif choice == "Go Back":
                    os.system('cls')
                    print("Going back...\n")

            elif choice == 'Manage teacher accounts':  # managing teacher accounts
                os.system('cls')
                while True:
                    print("Managing teacher accounts\n")
                    choice = questionary.select("Choices: ", choices=["Add a teacher account",
                                                                      "Assign teacher to subject",
                                                                      "Delete a teacher account",
                                                                      "Unassign teacher from subject",
                                                                      "Go Back"]).ask()

                    if choice == "Add a teacher account":  # adding a teacher account
                        admin.add_teacher()

                    elif choice == "Assign teacher to subject":  # assign teacher to a subject
                        admin.assign_teacher()

                    elif choice == "Delete a teacher account":  # deleting a teacher account
                        admin.delete_teacher()

                    elif choice == "Unassign teacher from subject":  # unassign teacher or TA from subject
                        admin.unassign_teacher()

                    elif choice == "Go Back":
                        os.system('cls')
                        print("Going back...\n")
                        break

            elif choice == 'Manage student accounts':  # managing student account/records
                os.system('cls')
                while True:
                    print("Managing student accounts/records\n")
                    choice = questionary.select("Choices: ", choices=["Create new student records for new batch",
                                                                      "Add a student account",
                                                                      "Delete a student account",
                                                                      "Go Back"]).ask()

                    # creating a new student records table for new batch and grade sheets
                    if choice == "Create new student records for new batch":
                        admin.new_studentbatch()

                    elif choice == "Add a student account":  # adding students to a record
                        admin.add_student()

                    elif choice == "Delete a student account":  # deleting a student from a record
                        admin.delete_student()

                    elif choice == "Go Back":
                        os.system('cls')
                        print("Going back...\n")
                        break

            elif choice == "Manage my account details":  # checking account details for this admin account
                admin.manage_account(user)

            elif choice == "Add admin account":  # add an admin account
                admin.add_admin(user)

            elif choice == 'Logout':
                os.system('cls')
                print("Logging out...\n")
                break

    def admin_auth():
        print("Admin Login\n")
        cursor.execute("SELECT username, password FROM courses_faculty.admins")
        login_info = dict()
        for x in cursor.fetchall():
            login_info[x[0]] = x[1]

        valid_login = False
        user = input("Username: ")
        passw = stdiomask.getpass(prompt="Password: ")

        if user in login_info.keys():
            if login_info[user] == passw:
                valid_login = True
                os.system('cls')
                admin.admin_session(user)

        if valid_login == False:
            os.system('cls')
            print("Invalid admin login details...\n")


'''-------------------------------------------------------------------------------------------------------------------------------------------'''


class start:
    def course_select():  # accessing tables in database courses to select a course to access
        cursor.execute("SHOW DATABASES LIKE 'courses_faculty'")
        dbs = [x[0] for x in cursor.fetchall()]

        if len(dbs) < 1:  # if no database 'courses' is found, creating it and adding necessary tables with initial values for them
            print("No courses_faculty database found, creating one...\n")
            # creating courses database if it doesn't exist
            cursor.execute("CREATE DATABASE courses_faculty")

            cursor.execute("USE courses_faculty")

            # creating admins table
            cursor.execute("CREATE TABLE IF NOT EXISTS admins (id INT AUTO_INCREMENT PRIMARY KEY, first_name TEXT DEFAULT NULL, last_name TEXT DEFAULT NULL, username VARCHAR(20) UNIQUE NOT NULL, password VARCHAR(20) NOT NULL, email TEXT NOT NULL, mobile VARCHAR(15) DEFAULT NULL, added_by VARCHAR(20) DEFAULT NULL)")

            print("Admins table created, input details for admin to be added: \n")
            while True:
                try:
                    first_name = input("Enter first name : ")
                    last_name = input("Enter last name : ")
                    username = input("Enter username : ")
                    password = input("Enter password for admin : ")
                    email = f"{username}_admin@dypiu.ac.in"
                    cursor.execute(
                        f"INSERT INTO admins (first_name, last_name, username, password, email) VALUES ('{first_name}', '{last_name}', '{username}', '{password}', '{email}')")

                    print(f"{username} added as an admin")
                    break
                except:
                    print("Enter appropriate values for each field...\n")

            # creating table in courses database to store courses
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS courses (id VARCHAR(10) NOT NULL, name TEXT, founded INT, years INT, type VARCHAR(10), PRIMARY KEY(id))")

            # adding a course into the table
            print("\nCourses table created, input details for a course in courses table: \n")
            while True:
                try:
                    id_ = input("Enter course id : ").replace(' ', '_')
                    name = input("Full name of course : ")
                    founded = int(input("What year was the course founded? : "))
                    years = int(input("How many years does the course last? : "))
                    course_type = questionary.select("What type of course is this?", choices=["Bachelors",
                                                                                              "Masters",
                                                                                              "PhD"]).ask()

                    cursor.execute(
                        f"INSERT INTO courses_faculty.courses VALUES ('{id_}', '{name}', {founded}, {years}, '{course_type}')")

                    print(f"{id_}, {name} successfully added\n")
                    break
                except:
                    print("Enter appropriate values for each field...\n")

            cursor.execute("CREATE TABLE IF NOT EXISTS teachers (teacher_id INT(5) NOT NULL AUTO_INCREMENT, first_name VARCHAR(30) NOT NULL, last_name VARCHAR(30) NOT NULL, username VARCHAR(20) NOT NULL, email TEXT(50) NULL, password VARCHAR(20) NOT NULL, type VARCHAR(20) NOT NULL DEFAULT 'Teacher', PRIMARY KEY (teacher_id), UNIQUE (username))")

            input("courses_faculty database created, press anything to continue...\n")
        os.system('cls')

        # accessing course ids from courses table
        cursor.execute("SELECT id FROM courses_faculty.courses")
        courses = [x[0] for x in cursor.fetchall()]
        courses.append("Exit")

        # choosing course
        print("Select course:\n")
        course = questionary.select("Choice : ", choices=courses).ask()

        if course != "Exit":
            cursor.execute(f"SELECT id, name FROM courses_faculty.courses WHERE id = '{course}'")
            # return needed details of the course for data manipulation
            return cursor.fetchall()[0]
        else:
            db.close()
            exit("Bye! ")

    def prerequisite_tables(database):
        # creating database for course if it doesnt exist already
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")

        cursor.execute(f"USE {database}")

        # creating prerequisite tables needed for the system to function if they dont exist already

        # creating foreign key for subjects and teachers for teacher_id
        cursor.execute("SHOW TABLES LIKE 'subjects'")
        if len([x[0] for x in cursor.fetchall()]) < 1:
            cursor.execute("CREATE TABLE IF NOT EXISTS subjects (id VARCHAR(10) NOT NULL , name TEXT NOT NULL ,semester INT(2) NOT NULL , teacher_id INT(5) DEFAULT NULL, ta_id INT(5) DEFAULT NULL, PRIMARY KEY (id))")

            cursor.execute(
                "ALTER TABLE subjects ADD FOREIGN KEY (teacher_id) REFERENCES courses_faculty.teachers(teacher_id) ON DELETE SET NULL ON UPDATE CASCADE")

            cursor.execute("ALTER TABLE subjects ADD FOREIGN KEY (ta_id) REFERENCES courses_faculty.teachers(teacher_id) ON DELETE SET NULL ON UPDATE CASCADE")

        # creating table to store students semester count, year start and expected year of graduation
        cursor.execute("CREATE TABLE IF NOT EXISTS students_batch (start_year INT NOT NULL AUTO_INCREMENT, grad_year INT, students INT NOT NULL DEFAULT 0, lat_students INT DEFAULT 0, tot_students INT DEFAULT 0, cur_semester INT DEFAULT 1, PRIMARY KEY (start_year))")

    def mainframe():
        os.system('cls')
        database, course = map(str, start.course_select())

        # checking if prerequisite tables for the database are there, if not, create them
        start.prerequisite_tables(database)

        """ PATH TO SQL DATABASES STORED ON YOUR DEVICE """
        global path
        path = f"C:\ProgramData\MySQL\MySQL Server 8.0\Data\{database}"

        while True:
            choice = questionary.select(f"{course} Login Page", choices=["Admin Login",
                                                                         "Teacher Login",
                                                                         "Student Login",
                                                                         "Exit"]).ask()

            if choice == 'Admin Login':
                os.system('cls')
                admin.admin_auth()

            elif choice == 'Teacher Login':
                os.system('cls')
                teacher().teacher_auth()

            elif choice == 'Student Login':
                os.system('cls')
                student().student_auth()

            else:
                start.course_select()


# war begins
if __name__ == "__main__":
    # initializing connection to MySQL
    db = mysql.connect(host='localhost', user='Ashwin', password='3431', autocommit=True, buffered=True)
    cursor = db.cursor()
    start.mainframe()
