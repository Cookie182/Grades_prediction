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

from scipy import stats as ss  # to generate numbers from a given range of numbers with a Gaussian weight distribution
from scipy.signal import savgol_filter  # add Savozky-Golay filler data to smoothen out lines on graphs

plt.style.use('bmh')  # matplotlib graph style
sns.set_style('dark')  # seaborn graph style
warnings.filterwarnings('ignore')  # to ignore messages from seaborn graphs

"""
    "CERTAIN CAVEATS:"
        "- IT IS HARDCODED THAT THERE ARE 2 SEMESTER PER YEAR, ONE STARTING AT JANUARY AND ANOTHER AT AUGUST"
        "- ALSO THE GRADE STRUCTURE IS HARD CODED TO:"
                                                    "* A+ = (>= 90%)"
                                                    "* A = (>= 80%)"
                                                    "* B = (>= 70%)"
                                                    "* C = (>= 60%) passing threshold"
                                                    "* D = (>= 50%)"
                                                    "* F = (< 50%)"
        ""- BEFORE ANY OPERATIONS, A DATABASE MUST BE CREATED VIA SQL, THE NAME OF THE DATABASE SHOULD BE THE NAME OF THE COURSE""
        "- THE 3 MAIN PREQUISITE TABLES, AS SHOWN IN THE FIRST OPTION IN THE ADMIN MENU (IT IS AUTOMATICALLY DONE AS SOON AS THE OPTION IS SELECTED TO RUN), ARE A MUST HAVE REQUIREMENT BEFORE CONDUCTING ANY TYPE OF OPERATIONS"
"""

""" MAKE SURE THE DATABASE DETAILS, SQL DETAILS AND PATH DETAILS ARE CORRECT FOR YOU NEAR THE END"""


def grades(test_type, test_amount, max_mark, weightage, pass_percent, final_test_name, n=1000, graphs=False):  # grades generator
    """Func that generates train/test data for the classifier/regressor and returns the trained classifier/regressor"""
    df = pd.DataFrame(index=range(1, n+1))  # making the dataframe and generating dummy marks
    df.index.name = 'Student'
    print("\nGenerating mock data\n")

    passfail_final, overallgrade_final = None, None
    passfail_final_acc, overallgrade_final_acc = 0, 0
    for test_run in range(1, 6):  # generating mock data 5 times to find the models with the higehst accuracy
        print(f"\nTest Run {test_run}\n")
        for x in range(len(test_type)):
            m = max_mark[x]  # storing max marks for each type of test
            if test_amount[x] > 1:  # generating random marks in marking range with a gaussian weight distribution to each mark
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
        passfail = make_pipeline(StandardScaler(),
                                 LogisticRegressionCV(Cs=np.arange(0.1, 1.1, 0.1),
                                                      cv=RepeatedStratifiedKFold(n_splits=10, random_state=7),
                                                      max_iter=1000, n_jobs=-1, refit=True,
                                                      random_state=7,
                                                      class_weight='balanced')).fit(X_test, y_test['Pass/Fail'])

        # final overall grade predictor
        overallgrade = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1)).fit(X_test, y_test['Total %'])
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

    if graphs == True:  # if explicitly stated to show the graphs of the distribution of grades
        for x in range(len(test_type)):
            if test_amount[x] > 1:  # plotting grade distribution for tests with more than one evaluation of that type
                if test_amount[x] % 2 != 0:
                    y = test_amount[x]+1
                else:
                    y = test_amount[x]

                fig = plt.figure(figsize=(10, 7), constrained_layout=True)
                grid = gs(nrows=int(y/2), ncols=2, figure=fig)
                for y in range(test_amount[x]):
                    ax = fig.add_subplot(grid[y])
                    sns.distplot(df.filter(regex=test_type[x]).iloc[:, y], fit=ss.norm, ax=ax, norm_hist=True, color='red',
                                 hist_kws=dict(edgecolor='black', align='right', color='red'),
                                 bins=max_mark[x]//2)
                    plt.xticks(size=10)
                    plt.yticks(size=12)
                    plt.xlabel('Marks', fontdict={'size': 13})
                    plt.ylabel('Density', fontdict={'size': 13})
                    plt.title(f"{test_type[x]} {y+1}", fontsize=14)
                    plt.tight_layout()
                fig.suptitle(f"Grade for {test_type[x]}", fontsize=15)
                grid.tight_layout(fig)
                if graphs == True:
                    plt.show()
            else:  # plotting grade distribution for singular evaluation test
                plt.figure(figsize=(8, 5))
                sns.distplot(df[test_type[x]], fit=ss.norm, norm_hist=True, color='red',
                             hist_kws=dict(edgecolor='black', align='right', color='red'),
                             bins=max_mark[x]//2)
                plt.title(f"Grade for {test_type[x]}", fontsize=14)
                plt.xlabel('Marks', fontdict={'size': 13})
                plt.xticks(size=12)
                plt.yticks(size=12)
                plt.ylabel('Density', fontdict={'size': 13})
                plt.tight_layout()
                plt.show()

        fig, ax = plt.subplots()
        plot_confusion_matrix(passfail, X_test, y_test['Pass/Fail'], labels=[0, 1],
                              display_labels=['Fail', 'Pass'], cmap='afmhot', ax=ax)
        plt.rcParams.update({'font.size': 18})
        ax.set_title('Confusion Matrix')
        plt.show()
        print(' ')

    return passfail_final, overallgrade_final


def rolling_predict(marks, subject, record):  # to present rolling predictions based on a student's marks
    all_marks = marks  # saved to calculate rolling actual grade (calculated with subject structure details)
    marks = np.array(marks[:-1]).reshape(1, -1)  # prepping data to be used for predictions
    # loading subject prediction models
    passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
    overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

    dummy = [0] * len(marks[0])  # making dummy list to cummulatively add each test score
    pass_probabs = []  # to store each probability as each test score gets entered
    for x in range(len(marks[0])):
        dummy[x] = marks[0][x]  # blink
        pass_probabs.append(passfail.predict_proba(np.array([dummy]))[0][1] * 100)

    # interpolating results to give a smoother graph
    pass_probabs_l = len(pass_probabs)
    if pass_probabs_l % 2 == 0:
        pass_probabs_l -= 1
    pass_probabs = savgol_filter(pass_probabs, pass_probabs_l, 4)

    limit1 = math.ceil(max([abs(x - 50) for x in pass_probabs]))  # limits determiend to scale the pass/fail graph better

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

    limit2 = math.ceil(max([abs(x-60) for x in total_percent]))  # limits determined to scale the overall grade graph better

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
        mults = [np.round(x*100, 2) for x in mults]
        p_grade = np.sum(mults)
        actual_grades.append(np.round(p_grade, 2))

    actual_calc_grade = actual_grades[-1]
    # interpolating results to give a smoother graph
    actual_grades_l = len(actual_grades)
    if actual_grades_l % 2 == 0:
        actual_grades_l -= 1

    actual_grades = savgol_filter(actual_grades, actual_grades_l, 4)

    limit3 = math.ceil(max([abs(x-60) for x in actual_grades]))  # limits determined to scale the overall grade graph better

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
    plt.suptitle(
        f"Chance of Passing and Predicted Total Grade for {subject}\nTake the predictions with a grain of salt", fontsize=12)
    ax1 = fig.add_subplot(grid[0, 0])
    plt.title(f"Probability of passing the subject after each test taken\nPredicted Pass or Fail? -> {pf}\
    \nChance of passing subject -> {passfail.predict_proba(marks)[0][1] * 100:.2f}%", fontsize=11)
    plt.axhline(50, color='r', label="Threshold", linestyle='--')
    plt.plot(tests[:-1], pass_probabs, c='black', lw=1, label='Predicted passing chance')
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8, rotation=45)
    plt.ylabel('Probability (%)', fontsize=9)
    plt.ylim(ymin=50-limit1, ymax=50+limit1)
    plt.margins(0.02, 0.02)
    plt.legend(loc='best', fontsize=7)
    plt.tight_layout()

    ax2 = fig.add_subplot(grid[0, 1])
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

    ax3 = fig.add_subplot(grid[1, :])
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


def student_session(record, user):
    os.system('cls')
    while True:
        cursor.execute(f"SELECT first_name FROM {record} WHERE username = '{user}'")
        print(f"Welcome to the student menu, {cursor.fetchall()[0][0]}\n")

        print("What to do?\n")
        choice = questionary.select("Choices: ", choices=["View grades",
                                                          "Show predictions with custom grades",
                                                          "View all subjects grades/details (including deleted ones)",
                                                          "View account details",
                                                          "Change account details",
                                                          "Logout"]).ask()
        if choice == "View grades":  # view grades of a subject
            os.system('cls')
            print("Viewing your grades\n")

            # getting semester choice to display subjects in said semester
            cursor.execute(f"SELECT cur_semester FROM {record} WHERE username = '{user}'")
            semesters = [str(x) for x in range(1, int(cursor.fetchall()[0][0])+1)]
            semesters.append("Go Back")

            print("Which semester?\n")
            semester = questionary.select("Choices: ", choices=semesters).ask()
            print(' ')

            if semester == "Go Back":
                os.system('cls')
                print("Going back from viewing grades...\n")
                continue

            # getting subjects that are in the specified semester
            cursor.execute(f"SELECT id, name FROM subjects WHERE semester = '{semester}'")
            subjects = [x[0] for x in cursor.fetchall()]
            subjects.append("Go Back")

            print("Which subject?\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            if subject != "Go Back":
                cursor.execute(f"SELECT id FROM {record} WHERE username = '{user}'")
                id = cursor.fetchall()[0][0]
                cursor.execute(f"DESCRIBE {record}_{subject}")
                tests = [x[0] for x in cursor.fetchall() if x[0] != 'student_id']
                print("")
                for test in tests:  # printing marks on a subject for each test
                    cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {id}")
                    print(f"{test} -> {cursor.fetchall()[0][0]}")

                marks = []  # storing marks of student

                print("What to do?\n")
                choice = questionary.select("Choices: ", choices=["View more info",
                                                                  "Go Back"]).ask()

                if choice == "View more info":  # to show prediction results graphs
                    cursor.execute(f"DESCRIBE {record}_{subject}")
                    tests = [x[0] for x in cursor.fetchall()][1:-3]
                    for test in tests:  # iterating and getting marks for tests except for the final test
                        cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {id}")
                        marks.append(cursor.fetchall()[0][0])
                    rolling_predict(marks, subject, record)
                    os.system('cls')
                    print(f"Marks displayed for {user} in {subject}\n")
                else:
                    os.system('cls')
                    print(f"Marks displayed for {user} in {subject}\n")
            else:
                os.system('cls')
                print("Going back...\n")
                continue

        elif choice == "Show predictions with custom grades":  # taking custom grades for a subject and showing predictions based on them
            os.system('cls')
            print("Show predictions with custom grades\n")

            # getting semester choice to display subjects in said semester
            cursor.execute(f"SELECT cur_semester FROM {record} WHERE username = '{user}'")
            semesters = [str(x) for x in range(1, int(cursor.fetchall()[0][0])+1)]
            semesters.append("Go Back")

            print("Which semester?\n")
            semester = questionary.select("Choices: ", choices=semesters).ask()
            print(' ')

            if semester == "Go Back":
                os.system('cls')
                print("Going back from viewing custom grades...\n")
                continue

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
                rolling_predict(custom_marks, subject, record)
                os.system('cls')
                print(f"Custom marks displayed for {user} in {subject}\n")

            else:
                os.system('cls')
                print("Going back...\n")
                continue

        elif choice == "View all subjects grades/details (including deleted ones)":  # showing marsheets for subjects that may have been deleted
            os.system('cls')
            while True:
                print("Showing all subjects for student's batch\n")
                cursor.execute(f"SHOW TABLES LIKE '{record}_%'")

                # getting subjects from the names of all the marksheets for the student's batch
                subjects = [str(x[0][len(record)+1:]).upper() for x in cursor.fetchall()]
                subjects.append("Go Back")

                print("View grades or subject details?\n")
                choice = questionary.select("Choices : ", choices=["View grades", "View subject details", "Go Back"]).ask()
                if choice == "View grades":
                    # getting subject choice
                    print("Which subject to view grade for?")
                    subject = questionary.select("Choice : ", subjects).ask()
                    if subject == "Go Back":
                        os.system('cls')
                        print("Going back...\n")
                        continue
                    else:
                        # getting column names
                        cursor.execute(f"DESC {record}_{subject}")
                        cols = [x[0] for x in cursor.fetchall()]

                        # getting data from marksheet for student
                        cursor.execute(f"SELECT * FROM {record}_{subject} WHERE student_id = (SELECT id FROM {record} WHERE username = '{user}')")
                        marks = cursor.fetchall()[0]

                        print(f"Showing your marks for {subject}\n")
                        [print(f"{cols[x]} -> {marks[x]}") for x in range(len(cols))]
                        input("Press anything to continue...\n")
                        os.system('cls')

                elif choice == "View subject details":  # viewing subject details taken by the student
                    print("Which subject to view details for?")
                    subject = questionary.select("Choice : ", subjects).ask()

                    if subject == "Go Back":
                        os.system('cls')
                        print("Going back...\n")
                        continue
                    else:
                        print(tabulate(pd.read_sql(f"SELECT * FROM {subject}_details", db, index_col='Type'), headers='keys', tablefmt='psql'))
                        print(f"\nDetails shown for {subject}\n")
                        input("Press anything to continue...\n")
                        os.system('cls')

                else:
                    os.system('cls')
                    break

        elif choice == "View account details":  # showing student account details
            os.system('cls')
            print("Showing account details\n")
            cursor.execute(f"DESCRIBE {record}")
            cols = [x[0] for x in cursor.fetchall()]
            for col in cols:
                cursor.execute(f"SELECT {col} FROM {record} WHERE username = '{user}'")
                print(f"{col} -> {cursor.fetchall()[0][0]}")

            input("\nPress anything to continue...")
            os.system('cls')
            print(f"Account details for username {user} shown\n")

        elif choice == "Change account details":  # changing detail in student account
            os.system('cls')
            print(f"Changing account details for {user}\n")
            cursor.execute(f"DESCRIBE {record}")
            cols = [x[0] for x in cursor.fetchall() if x[0] in ["first_name", "last_name", "mobile_no", "username", "password"]]
            cols.append("Go Back")

            print("Which detail to change?\n")
            col = questionary.select("Choices: ", choices=cols).ask()

            if col != "Go Back":
                while True:
                    try:
                        cursor.execute(f"SELECT {col} FROM {record} WHERE username = '{user}'")
                        old_detail = cursor.fetchall()[0][0]
                        new_detail = input(f"\nOld {col} -> {old_detail}\nEnter new {col} : ")
                        cursor.execute(f"UPDATE {record} SET {col} = '{new_detail}' WHERE username = '{user}'")
                        db.commit()
                        break
                    except:
                        print("Enter valid input...\n")

                # updating user variable incase student decides to update it
                cursor.execute(f"SELECT username FROM {record} WHERE {col} = '{new_detail}'")
                user = cursor.fetchall()[0][0]

                os.system('cls')
                print(f"Account detail {col} changed from {old_detail} to {new_detail}\n")
            else:
                os.system('cls')
                print("Going back...\n")
                continue

        elif choice == "Logout":
            os.system('cls')
            print("Logging out from student menu\n")
            break


def student_auth():
    os.system('cls')
    print("Student Login")
    cursor.execute("SELECT start_year FROM students_batch")
    records = [str(x[0]) for x in cursor.fetchall()]
    records.append("Go Back")
    print("\nWhich record do you belong to?\n")
    record = questionary.select("Choices: ", choices=records).ask()  # choosing which student record they belong to
    if record != "Go Back":
        record = f"students_{record}"
        user = input("Username : ")  # checking login details
        passw = stdiomask.getpass(prompt='Password : ')

        valid_login = False
        cursor.execute(f"SELECT username FROM {record}")  # gathering all the student usernames from selected student record
        if user in [x[0] for x in cursor.fetchall()]:
            cursor.execute(f"SELECT password FROM {record} WHERE username = '{user}'")
            if passw == cursor.fetchall()[0][0]:

                valid_login = True
                student_session(record, user)  # launching student session

        if valid_login == False:
            os.system('cls')
            print("Incorrect student login details...\n")

    else:
        os.system('cls')
        print("Incorrect choice for student record, going back...\n")


def teacher_session(teacher_id):
    os.system('cls')
    while True:
        cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = {teacher_id}")
        subjects = [x[0] for x in cursor.fetchall()]  # keeping a track of this particular teacher's
        subjects.append("Go Back")
        cursor.execute(f"SELECT first_name FROM teachers WHERE teacher_id = '{teacher_id}'")
        first_name = cursor.fetchall()[0][0]
        print(f"Welcome to the teacher menu, {first_name}\n")

        print("What to do?\n")
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
            os.system('cls')
            print("Review overall marks for a subject\n")

            print("Which subject?\n")
            choice = questionary.select("Choices: ", choices=subjects).ask()

            # choosing for which subject to edit marks
            if choice != "Go Back":
                print("")
                subject = choice
                cursor.execute(f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id = '{subject}')")
                # choosing for which batch to edit marks for
                records = [str(x[0]) for x in cursor.fetchall()]
                records.append("Go Back")

                print("Which batch?\n")
                record = questionary.select("Choices: ", choices=records).ask()
                if len(records) < 1:
                    os.system('cls')
                    print("No student batch have reached this subject yet...\n")
                    continue

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
                print("Incorrect subject choice, going back...\n")
                continue

        elif choice == "Update a test score":  # updating scores of students for a test
            os.system('cls')
            print("Updating a test score\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            # choosing for which subject to edit marks
            if subject != "Go Back":
                print("")
                cursor.execute(f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id='{subject}')")
                # choosing for which batch to edit marks for
                records = [str(x[0]) for x in cursor.fetchall()]
                if len(records) < 1:
                    os.system('cls')
                    print("No student batch has reached this subject yet....\n")
                    continue

                records.append("")

                print("Choose student batch\n")
                record = questionary.select("Choices: ", choices=records).ask()

                if record != "Go Back":  # choosing student record
                    record = f"students_{record}"
                    cursor.execute(f"SELECT type FROM {subject}_details")
                    print("Which test type?: ")
                    type = questionary.select("Choices: ", choices=[x[0] for x in cursor.fetchall()]).ask()

                    print(' ')
                    cursor.execute(f"SELECT amount FROM {subject}_details WHERE type = '{type}'")  # getting amount for that test type
                    amount = int(cursor.fetchall()[0][0])

                    if amount == 1:
                        test = type
                    else:
                        print("Which test?\n")
                        test = questionary.select("Choices: ", choices=[f"{type}_{x}" for x in range(1, amount+1)]).ask()

                    # loading subject prediction models
                    passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
                    overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

                    os.system('cls')
                    print(f"Editing marks of {test} for {record} for {subject}\n")
                    # getting the list of student_ids in this mark sheet
                    cursor.execute(f"SELECT Max_Mark FROM {subject}_details WHERE type = '{type}'")
                    max_mark = int(cursor.fetchall()[0][0])

                    cursor.execute(f"SELECT student_id FROM {record}_{subject}")
                    ids = [x[0] for x in cursor.fetchall()]  # iterating for each student
                    print("Enter marks\n")
                    for id in ids:
                        while True:
                            try:
                                mark = int(input(f"Student {id} -> "))
                                # update mark of each student
                                if mark > max_mark:
                                    print(f"Enter a valid mark for {type}...\n")
                                    continue

                                cursor.execute(f"UPDATE {record}_{subject} SET {test} = {mark} WHERE student_id = {id}")
                                db.commit()
                                break
                            except:
                                print(f"Enter a valid mark for {type}...\n")

                    for id in ids:
                        cursor.execute(f"DESCRIBE {record}_{subject}")
                        # get the name of tests for prediction
                        tests = [x[0] for x in cursor.fetchall()][1:-4]
                        mark = []
                        for test in tests:
                            cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {id}")
                            mark.append(cursor.fetchall()[0][0])

                        # converting to a 2d array so scikit-learn models can use them for predictions
                        mark = np.array(mark).reshape(1, -1)
                        predict_passing = round(passfail.predict_proba(mark)[0][1]*100, 2)  # proabability of passing the subject
                        predict_grade = round(overallgrade.predict(mark)[0]*100, 2)  # predicted grade of overall subject after finals
                        predict_lettergrade = None  # giving letter based grade based on predicted overall percentage after finals
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

                        cursor.execute(f"UPDATE {record}_{subject} SET PASS_CHANCE = {predict_passing} WHERE student_id = {id}")
                        db.commit()

                        cursor.execute(f"UPDATE {record}_{subject} SET PREDICTED_GRADE = '{predict_grade}' WHERE student_id = {id}")
                        db.commit()

                        # gathering tests to calculate actual grade using subject details and actual current marks
                        cursor.execute(f"SELECT * FROM {record}_{subject} WHERE student_id = {id}")
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
                        cursor.execute(f"UPDATE {record}_{subject} SET GRADE = '{grade}' WHERE student_id = {id}")
                        db.commit()

                    os.system('cls')
                    print(f"Marks for {test} in {subject} updated\n")

                else:
                    os.system('cls')
                    print("Incorrect option for student batch, going back...\n")
                    continue

            elif int(choice) == len(subjects)+1:
                os.system('cls')
                print("Going back...\n")
                continue

        elif choice == "Update a student's score":  # updating grade of one student in a particular subject
            os.system('cls')
            print("Update grade for a student\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            # choosing for which subject to edit marks
            if subject != "Go Back":
                print("")
                cursor.execute(f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id = '{subject}')")
                # choosing for which batch to edit marks for
                record = [str(x[0]) for x in cursor.fetchall()]
                if len(record) < 1:
                    os.system('cls')
                    print("No student batch has reached this subject yet...\n")
                    continue
                record.append("Go Back")

                print("Choose student batch\n")
                choice = questionary.select("Choice: ", choices=record).ask()

                if choice != "Go Back":
                    record = f"students_{choice}"
                    print("\nChoose student id to edit marks for : \n")
                    cursor.execute(f"SELECT student_id FROM {record}_{subject}")  # choosing student
                    ids = [str(x[0]) for x in cursor.fetchall()]
                    ids.append("Go Back")

                    print("Choose student id\n")
                    student_id = questionary.select("Choices: ", choices=ids).ask()

                    if student_id != "Go Back":
                        cursor.execute(f"SELECT type FROM {subject}_details")
                        type = questionary.select("Choices: ", choices=[x[0] for x in cursor.fetchall()]).ask()

                        print(' ')
                        cursor.execute(f"SELECT amount FROM {subject}_details WHERE type = '{type}'")  # getting amount for that test type
                        amount = int(cursor.fetchall()[0][0])

                        if amount == 1:
                            test = type
                        else:
                            print("Which test?\n")
                            test = questionary.select("Choices: ", choices=[f"{type}_{x}" for x in range(1, amount+1)]).ask()

                        mark = input(f"\nEnter new mark for student_id {student_id} in {test} for {subject} : ")
                        cursor.execute(f"UPDATE {record}_{subject} SET {test} = {int(mark)} WHERE student_id = {student_id}")
                        db.commit()

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
                        predict_passing = round(passfail.predict_proba(mark)[0][1]*100, 2)  # proabability of passing the subject
                        predict_grade = round(overallgrade.predict(mark)[0]*100, 2)  # predicted grade of overall subject after finals
                        predict_lettergrade = None  # giving letter based grade based on predicted overall percentage after finals
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
                        db.commit()

                        cursor.execute(f"UPDATE {record}_{subject} SET PREDICTED_GRADE = '{predict_grade}' WHERE student_id = {student_id}")
                        db.commit()
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
                        db.commit()

                        os.system('cls')
                        print(f"Mark updated for student_id {student_id} in {test} for {subject}\n")

                    else:
                        os.system('cls')
                        print("Incorrect choice for student id, going back...\n")
                else:
                    os.system('cls')
                    print("Incorrect choice for student batch, going back...\n")
            else:
                os.system('cls')
                print("Incorrect choice for subject selection, going back...\n")

        elif choice == "List Passing/Failing students":  # list the students who are actually and predicted to passing/failing
            os.system('cls')
            print("Listing students who are actually and predicted to pass/fail\n")
            cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = '{teacher_id}'")
            subjects = [x[0] for x in cursor.fetchall()]
            subjects.append("Go Back")

            print("Which subject?\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            if subject != "Go Back":
                cursor.execute(f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id = '{subject}')")
                records = [str(x[0]) for x in cursor.fetchall()]
                if len(records) < 1:
                    os.system('cls')
                    print("There are no student batches who are in the semester (or after) this subject...\n")
                    continue

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
                    elif choice == "Calculated Grades":  # actual grades (so far)
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

        elif choice == "View students' performance":  # visually interpreting students' performance in a subject
            os.system('cls')
            print("Visually interpreting students' performance for a subject\n")
            cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = '{teacher_id}'")
            subjects = [x[0] for x in cursor.fetchall()]
            subjects.append("Go Back")

            print("Which subject?\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            if subject != "Go Back":
                cursor.execute(f"SELECT start_year FROM students_batch WHERE cur_semester >= (SELECT semester FROM subjects WHERE id='{subject}')")
                records = [str(x[0]) for x in cursor.fetchall()]
                if len(records) < 1:
                    os.system('cls')
                    print("There are no student batches who are in the semester (or after) this subject...\n")
                    continue

                print("Which batch?\n")
                record = questionary.select("Choices: ", choices=records).ask()
                # getting the predicted and actual grades
                df = pd.read_sql(f"SELECT * FROM students_{record}_{subject}",
                                 db, index_col='student_id')[['PREDICTED_GRADE', 'GRADE']]

                try:
                    # prediction grades
                    pred_grade_p = np.array([1 if x >= 60 else 0 for x in [float(x.split()[0]) for x in df['PREDICTED_GRADE']]])  # numerical grade
                    pred_grade_l = np.array([x.split()[1][1] for x in df['PREDICTED_GRADE']])  # letter grade

                    # actual grades (manually calculated)
                    actual_grade_p = np.array([1 if x >= 60 else 0 for x in [float(x.split()[0]) for x in df['GRADE']]])
                    actual_grade_l = np.array([x.split()[1][1] for x in df['GRADE']])
                except:
                    os.system('cls')
                    print("No predicted grades available, going back...\n")
                    continue

                fig = plt.figure()
                # predictions
                fig.add_subplot(221)
                plt.title("Predicted Pass/Fail", fontsize=12)
                plt.pie([np.count_nonzero(pred_grade_p == 1), np.count_nonzero(pred_grade_p == 0)], labels=['Pass', 'Fail'], autopct='%.2f%%')
                plt.legend(loc='best')
                plt.tight_layout()

                fig.add_subplot(222)
                plt.title("Predicted Letter Grades", fontsize=12)
                plt.pie([np.count_nonzero(pred_grade_l == x) for x in np.unique(pred_grade_l)], labels=np.unique(pred_grade_l), autopct='%.2f%%')
                plt.legend(loc='best')
                plt.tight_layout()

                # actual grades
                fig.add_subplot(223)
                plt.title("Actual Pass/Fail", fontsize=12)
                plt.pie([np.count_nonzero(actual_grade_p == 1), np.count_nonzero(actual_grade_p == 0)], labels=['Pass', 'Fail'], autopct='%.2f%%')
                plt.legend(loc='best')
                plt.tight_layout()

                fig.add_subplot(224)
                plt.title("Actual Letter Grades", fontsize=12)
                plt.pie([np.count_nonzero(actual_grade_l == x) for x in np.unique(actual_grade_l)], labels=np.unique(actual_grade_l), autopct='%.2f%%')
                plt.legend(loc='best')
                plt.tight_layout()
                plt.show()

                os.system('cls')
                print(f"Student's performance shown for {subject}\n")

            else:
                os.system('cls')
                print("Incorrect subject choice, going back...\n")

        elif choice == "Edit account info":  # changing first_name, last_name or password of teacher account
            os.system('cls')
            print("Changing teacher account details\n")
            df = pd.read_sql(f"SELECT * FROM teachers WHERE teacher_id = {teacher_id}", db, index_col='teacher_id')
            print(tabulate(df, headers='keys', tablefmt='psql'), '\n')
            cursor.execute("DESCRIBE teachers")
            details = [x[0] for x in cursor.fetchall()]

            for x in ['teacher_id', 'email']:  # remove details of account that cannot be changed
                details.remove(x)

            details.append("Go Back")
            print("Which detail to change?")
            detail = questionary.select("Choose: ", choices=details).ask()

            if detail != "Go Back":
                print(f"\nChanging {detail}\n")
                new_detail = input(f"Enter new {detail} : ")
                cursor.execute(f"UPDATE teachers SET {detail} = '{new_detail}' WHERE teacher_id = {teacher_id}")
                db.commit()
                os.system('cls')
                print(f"Account detail for {detail} updated\n")

            else:
                os.system('cls')
                print("Incorrect choice for account detail to change, going back...\n")
                continue

        elif choice == "View subject details":  # viewing subject details that the teacher teaches
            os.system('cls')
            print("Viewing subject details\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            if subject != "Go Back":
                table = f"{subject}_details"
                print(tabulate(pd.read_sql(f"SELECT * FROM {table}", db, index_col="Type"), headers='keys', tablefmt='psql'))
                print(f"Details for {subject} shown\n")
                input("\nEnter anything to continue...")
                os.system('cls')
            else:
                os.system('cls')
                print("Invalid option for subject choice, going back\n")

        elif choice == "Change subject details":  # changing subject details
            os.system('cls')
            print("Changing subject details\n")
            print("Which subject?\n")
            subject = questionary.select("Choices: ", choices=subjects).ask()

            if subject != "Go Back":
                print(f"Are you sure you want to modify {subject}?\n")
                _ = questionary.select("Choices: ", choices=["Yes", "No"]).ask()
                if _ == "No":
                    os.system('cls')
                    continue

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
                    test_amount = tuple(int(input(f"How many tests for {x}?: ")) for x in test_type)  # amount of tests per evaluation
                    print(' ')
                    break
                except:
                    print("Enter valid inputs for test amounts of each test type...\n")

            # max mark for each type of evaluation
            while True:
                try:
                    max_mark = tuple(int(input(f"{x} out of how many marks?: "))
                                     for x in test_type)  # maximum marks for each type of tests
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
            cursor.execute(f"DROP TABLE {subject}_details")
            db.commit()

            cursor.execute(
                f"CREATE TABLE {table_name} (Type VARCHAR(30), Amount INT(2), Weightage FLOAT, Max_mark INT(3))")
            db.commit()

            passfail, overallgrade = grades(test_type, test_amount, max_mark,
                                            weightage, pass_percent, final_test_name)

            with open(f"{path}/{subject}_passfail", 'wb') as f:
                pickle.dump(passfail, f)

            with open(f"{path}/{subject}_overallgrade", 'wb') as f:
                pickle.dump(overallgrade, f)

            # inserting details about new subject
            for x in [tuple((test_type[x], test_amount[x], weightage[x], max_mark[x])) for x in range(len(test_type))]:
                cursor.execute(f"INSERT INTO {table_name} VALUES ('{x[0]}', {x[1]}, {x[2]}, {x[3]})")
                db.commit()

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
            cursor.execute(f"SELECT start_year FROM students_batch WHERE cur_semester <= (SELECT semester FROM subjects WHERE id = '{subject}')")
            for record in [int(x[0]) for x in cursor.fetchall()]:
                marksheet = f"students_{record}_{subject}"

                # storing the old marksheet
                df = pd.read_sql(f"SELECT * FROM {marksheet}", db, index_col='student_id')

                # dropping columns other than student_id col on the marksheet (to make room to add new test names)
                cursor.execute(f"DESC {marksheet}")
                for old_col in [x[0] for x in cursor.fetchall() if x[0] != 'student_id']:
                    cursor.execute(f"ALTER TABLE {marksheet} DROP COLUMN {old_col}")
                    db.commit()

                # adding new tests to marksheet
                for new_col in tests:
                    cursor.execute(f"ALTER TABLE {marksheet} ADD COLUMN {new_col} INT(3) NOT NULL DEFAULT 0")
                    db.commit()

                # adding prediction columns and grade column
                cursor.execute(f"ALTER TABLE {marksheet} ADD PASS_CHANCE FLOAT NOT NULL DEFAULT 0")
                db.commit()
                cursor.execute(f"ALTER TABLE {marksheet} ADD PREDICTED_GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                db.commit()
                cursor.execute(f"ALTER TABLE {marksheet} ADD GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                db.commit()

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
                            db.commit()

                    os.system('cls')
                    print(f"Details for {marksheet} added\n")

        elif choice == "Logout":  # logging out of teacher session
            os.system('cls')
            print("Logging out of teachers session...\n")
            break


def teacher_auth():
    print("Teacher Login\n")
    user = input("Username : ")
    passw = stdiomask.getpass(prompt='Password : ')

    # checking if both username AND password match respectively
    valid_login = False
    cursor.execute("SELECT username FROM teachers")  # gathering all the teacher usernames
    if user in [x[0] for x in cursor.fetchall()]:
        cursor.execute(f"SELECT password FROM teachers WHERE username = '{user}'")
        if passw == cursor.fetchall()[0][0]:
            cursor.execute(f"SELECT teacher_id FROM teachers WHERE username = '{user}'")
            valid_login = True
            teacher_session(cursor.fetchall()[0][0])  # launching teacher session for teacher

    if valid_login == False:
        os.system('cls')
        print("Incorrect teacher login details...\n")


def admin_session(user):
    while True:
        print(f"Welcome to the admin menu, {user}\n")

        print("What to do?\n")
        choice = questionary.select("Choices: ", choices=["Show tables",
                                                          "Manage subjects",
                                                          "Manage courses",
                                                          "Manage teacher accounts",
                                                          "Manage student accounts",
                                                          "Manage my account details",
                                                          "Add admin account",
                                                          "Logout"]).ask()

        if choice == 'Show tables':  # showing all table names and details about them
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

        elif choice == "Manage courses":  # adding or modifying courses
            os.system('cls')
            while True:
                print("Managing courses\n")

                choice = questionary.select("Choices: ", choices=["Add a course",
                                                                  "Go Back"]).ask()
                if choice == "Add a course":
                    print("\nAdding a course\n")
                    while True:
                        try:
                            id = input("Enter course id : ").replace(' ', '_')
                            name = input("Full name of course : ")
                            founded = int(input("What year was the course founded? : "))
                            length = int(input("How many years does the course last? : "))
                            department = input("Which department does this course belong to? : ")
                            department_head = input("Name of department head : ")

                            cursor.execute(
                                f"INSERT INTO courses.courses VALUES ('{id}', '{name}', {founded}, {length}, '{department}', '{department_head}')")
                            db.commit()
                            print(f"{id}, {name} successfully added\n")
                            input("Press anything to continue")
                            os.system('cls')
                            break
                        except:
                            print("Enter appropriate values for each field...\n")

                else:
                    os.system('cls')
                    print("Going back...\n")
                    break

        elif choice == 'Manage subjects':  # adding or deleting a subject
            os.system('cls')
            while True:
                print("Managing subjects\n")

                print("What to do?\n")
                choice = questionary.select("Choices: ", choices=["Add a subject",
                                                                  "Delete a subject",
                                                                  "Go Back"]).ask()
                if choice == 'Add a subject':
                    print("\nAdding a subject\n")
                    teachers = pd.read_sql("SELECT * FROM teachers", db, index_col='teacher_id')
                    if len(teachers) < 1:
                        print("There are no teachers registered in the records yet\n")
                        continue

                    print("IS THE TEACHER THAT TEACHES THIS SUBJECT IN THE TABLE?\n")
                    print(tabulate(teachers, headers='keys', tablefmt='psql'), '\n')
                    choice = questionary.select("Choices: ", choices=['Yes', 'No']).ask()
                    if choice == "Yes":
                        print(' ')
                        cursor.execute("SELECT id FROM subjects")
                        subject_names = [x[0] for x in cursor.fetchall()]
                        subj_name = input("Enter abbreviation of subject : ").strip().upper()

                        if subj_name not in subject_names:
                            try:
                                full_name = input("Enter full name of subject : ").strip()
                                semester = int(input("Which semester is this subject in : "))
                                teach = int(input("Enter teacher ID for this subject : "))
                                cursor.execute(f"INSERT INTO subjects VALUES ('{subj_name}', '{full_name}', {semester}, {teach})")
                                db.commit()

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
                                        test_amount = tuple(int(input(f"How many tests for {x}?: ")) for x in test_type)  # amount of tests per evaluation
                                        print(' ')
                                        break
                                    except:
                                        print("Enter valid inputs for test amounts of each test type...\n")

                                # max mark for each type of evaluation
                                while True:
                                    try:
                                        max_mark = tuple(int(input(f"{x} out of how many marks?: "))
                                                         for x in test_type)  # maximum marks for each type of tests
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

                                cursor.execute(
                                    f"CREATE TABLE {table_name} (Type VARCHAR(30), Amount INT(2), Weightage FLOAT, Max_mark INT(3))")
                                db.commit()

                                passfail, overallgrade = grades(test_type, test_amount, max_mark,
                                                                weightage, pass_percent, final_test_name)

                                with open(f"{path}/{subj_name}_passfail", 'wb') as f:
                                    pickle.dump(passfail, f)

                                with open(f"{path}/{subj_name}_overallgrade", 'wb') as f:
                                    pickle.dump(overallgrade, f)

                                # inserting details about new subject
                                for x in [tuple((test_type[x], test_amount[x], weightage[x], max_mark[x])) for x in range(len(test_type))]:
                                    cursor.execute(f"INSERT INTO {table_name} VALUES ('{x[0]}', {x[1]}, {x[2]}, {x[3]})")
                                    db.commit()

                                print(f"Details for {full_name} added\n")

                                # getting appropriate student record
                                cursor.execute(
                                    f"SELECT start_year FROM students_batch WHERE cur_semester <= {semester}")
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
                                            db.commit()

                                            # adding foreign key to link student ids together
                                            cursor.execute(
                                                f"ALTER TABLE {new_table} ADD FOREIGN KEY (student_id) REFERENCES {table}(id) ON DELETE CASCADE ON UPDATE CASCADE")
                                            db.commit()

                                            # adding columns of tests and making the default marks 0
                                            for test in tests:
                                                cursor.execute(f"ALTER TABLE {new_table} ADD {test} INT(3) NOT NULL DEFAULT 0")
                                                db.commit()

                                            # custom columns to store predictor variables
                                            cursor.execute(f"ALTER TABLE {new_table} ADD PASS_CHANCE FLOAT NOT NULL DEFAULT 0")
                                            db.commit()
                                            cursor.execute(f"ALTER TABLE {new_table} ADD PREDICTED_GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                                            db.commit()
                                            cursor.execute(f"ALTER TABLE {new_table} ADD GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                                            db.commit()

                                            # adding each student id for each student record table and this subject
                                            cursor.execute(f"SELECT id FROM {table}")
                                            for x in [x[0] for x in cursor.fetchall()]:
                                                cursor.execute(f"INSERT INTO {new_table} (student_id) VALUES ({x})")
                                                db.commit()

                                            cursor.execute(
                                                f"CREATE TRIGGER {new_table} AFTER INSERT ON {table} FOR EACH ROW INSERT INTO {new_table} (student_id) values (new.id)")
                                            db.commit()

                                    print(f"Grades sheets for {subj_name} created\n")
                                    input("Press anything to continue...")
                                    os.system('cls')
                                else:
                                    os.system('cls')
                                    print("No student record tables found...\n")
                            except:
                                cursor.execute(f"DELETE FROM subjects WHERE id = '{subj_name.upper()}'")
                                db.commit()

                                # selects appropriate batches to add ths subjects for
                                cursor.execute("SELECT start_year FROM students_batch")
                                for year in [int(x[0]) for x in cursor.fetchall()]:
                                    cursor.execute(f"DROP TRIGGER IF EXISTS students_{year}_{subj_name.upper()}")
                                    db.commit()

                                cursor.execute(f"SHOW TABLES LIKE '%{subj_name}%'")
                                for table in [x[0] for x in cursor.fetchall()]:
                                    cursor.execute(f"DROP TABLE {table}")
                                    db.commit()

                                os.system('cls')
                                print("Enter valid subject details...\n")
                        else:
                            os.system('cls')
                            print(f"{subj_name} already exists in the subjects table...\n")
                    else:
                        os.system('cls')
                        print("Make sure to enter the teacher details in the teacher table first...\n")

                elif choice == 'Delete a subject':  # deleting a subject
                    os.system('cls')
                    print("Deleting a subject\n")
                    cursor.execute("SELECT id FROM subjects")
                    subjects = [x[0] for x in cursor.fetchall()]
                    subjects.append("Go Back")

                    print("Which subject to delete?\n")
                    choice = questionary.select("Choices: ", choices=subjects).ask()

                    if choice == "Go Back":
                        os.system('cls')
                        print("Going back...\n")
                        continue
                    else:
                        subject = choice
                        print(f"\nAre you sure you want to delete {subject}?\n")
                        choice = questionary.select("Choices: ", choices=["Yes", "No"]).ask()

                        if choice == "Yes":
                            cursor.execute(f"DELETE FROM subjects WHERE id = '{subject}'")
                            db.commit()

                            # selecting appropriate student batches
                            cursor.execute(
                                f"SELECT start_year FROM students_batch WHERE cur_semester <= (SELECT semester FROM subjects where id = '{subject}')")
                            records = [f"students_{x[0]}" for x in cursor.fetchall()]
                            for record in records:
                                cursor.execute(f"DROP TABLE IF EXISTS {record}_{subject}")
                                db.commit()

                                cursor.execute(f"SHOW TABLES LIKE '%{subject}%'")
                                tables = [x[0] for x in cursor.fetchall()]
                                if len(tables) > 0:
                                    for table in tables:
                                        cursor.execute(f"DROP TABLE {table}")
                                        db.commit()

                                cursor.execute(f"DROP TRIGGER IF EXISTS {record}_{subject}")
                                db.commit()

                            if os.path.exists(f"{path}/{subject}_passfail"):
                                os.remove(f"{path}/{subject}_passfail")

                            if os.path.exists(f"{path}/{subject}_overallgrade"):
                                os.remove(f"{path}/{subject}_overallgrade")

                            os.system('cls')
                            print(f"{subject} has been deleted\n")

                        else:
                            os.system('cls')
                            print("Going back...\n")
                            break

                elif choice == "Go Back":
                    os.system('cls')
                    print("Going back...\n")
                    break

        elif choice == 'Manage teacher accounts':  # managing teacher accounts
            os.system('cls')
            while True:
                print("Managing teacher accounts\n")

                print("What to do?\n")
                choice = questionary.select("Choices: ", choices=["Add a teacher account",
                                                                  "Delete a teacher account",
                                                                  "Go Back"]).ask()

                if choice == "Add a teacher account":  # adding a teacher account
                    os.system('cls')
                    print("Adding a teacher\n")
                    first = input("First name : ")
                    last = input("Last name : ")
                    user = input("Enter username : ")
                    passw = input("Enter password : ")
                    email = f"{first}.{last}@dypiu.ac.in"
                    cursor.execute(
                        f"INSERT INTO teachers (first_name, last_name, username, email, password) VALUES ('{first}', '{last}', '{user}', '{email}', '{passw}')")
                    db.commit()
                    os.system('cls')
                    print(f"{user} has been added as a teacher\n")

                elif choice == "Delete a teacher account":  # deleting a teacher account
                    os.system('cls')
                    print("Deleting a teacher\n")
                    print(tabulate(pd.read_sql("SELECT * FROM teachers", db, index_col='teacher_id'), headers='keys', tablefmt='psql'), '\n')
                    cursor.execute("SELECT teacher_id FROM teachers")
                    print("Choose id of teacher to delete\n")
                    id = int(questionary.select("Choices: ", choices=[str(x[0]) for x in cursor.fetchall()]).ask())

                    print("Are you sure?\n")
                    choice = questionary.select("Choices: ", choices=["Yes", "No"]).ask()
                    if choice == "Yes":
                        cursor.execute(f"SELECT username FROM teachers WHERE teacher_id = '{id}'")
                        user = cursor.fetchall()[0][0]
                        cursor.execute(f"DELETE FROM teachers WHERE teacher_id = '{id}'")
                        db.commit()
                        os.system('cls')
                        print(f"{user} has been deleted from teacher records\n")
                    else:
                        os.system('cls')
                        print("Invalid choice, going back...\n")

                elif choice == "Go Back":
                    os.system('cls')
                    print("Going back...\n")
                    break

        elif choice == 'Manage student accounts':  # managing student account/records
            os.system('cls')
            while True:
                print("Managing student accounts/records\n")

                print("What to do?\n")
                choice = questionary.select("Choices: ", choices=["Create new student records for new batch",
                                                                  "Add a student account",
                                                                  "Delete a student account",
                                                                  "Go Back"]).ask()

                if choice == "Create new student records for new batch":  # creating a new student records table for new batch and grade sheets
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
                        # creating entry for students batch table
                        cursor.execute(f"INSERT INTO students_batch (start_year, grad_year) VALUES ({int(year)}, {int(year) + course_len})")
                        db.commit()

                        table = f"students_{year}"  # name of new student records
                        # creating student record for that batch
                        cursor.execute(
                            f"CREATE TABLE {table}(id INT(3) NOT NULL AUTO_INCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, mobile_no VARCHAR(15) DEFAULT NULL, email VARCHAR(40) DEFAULT NULL, username VARCHAR(20) UNIQUE NOT NULL, password VARCHAR(20) NOT NULL, start_year INT NOT NULL DEFAULT {int(year)},start_sem INT NOT NULL DEFAULT 1, cur_semester INT NOT NULL DEFAULT 1, entry VARCHAR(20) DEFAULT 'Normal', grad_year INT DEFAULT {int(year)+4}, PRIMARY KEY(id), UNIQUE(username))")
                        db.commit()

                        # foreign key referencing students_batch table to student record
                        cursor.execute(f"ALTER TABLE {table} ADD FOREIGN KEY (start_year) REFERENCES students_batch(start_year)")
                        db.commit()

                        # triggers for counting normal entry students after either deleting/inserting new data for each record
                        cursor.execute(
                            f"CREATE TRIGGER normal_student_count_{year}_insert AFTER INSERT ON {table} FOR EACH ROW UPDATE students_batch SET students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Normal') WHERE start_year={int(year)}")
                        db.commit()

                        cursor.execute(
                            f"CREATE TRIGGER normal_student_count_{year}_delete AFTER DELETE ON {table} FOR EACH ROW UPDATE students_batch SET students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Normal') WHERE start_year={int(year)}")
                        db.commit()

                        # triggers for counting lateral entry students after either deleting/inserting new data for each record
                        cursor.execute(
                            f"CREATE TRIGGER lateral_student_count_{year}_insert AFTER INSERT ON {table} FOR EACH ROW UPDATE students_batch SET lat_students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Lateral') WHERE start_year={int(year)}")
                        db.commit()

                        cursor.execute(
                            f"CREATE TRIGGER lateral_student_count_{year}_delete AFTER DELETE ON {table} FOR EACH ROW UPDATE students_batch SET lat_students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Lateral') WHERE start_year={int(year)}")
                        db.commit()

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
                            table_name = f"{subject}_details"  # name of subjects to iterate for making grade sheets
                            new_table = f"{table}_{subject}"  # name of table to store grades on subject for new students

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
                            db.commit()

                            # adding foreign key to link student ids together
                            cursor.execute(
                                f"ALTER TABLE {new_table} ADD FOREIGN KEY (student_id) REFERENCES {table}(id) ON DELETE CASCADE ON UPDATE CASCADE")
                            db.commit()

                            # adding columns of tests and making the default marks 0
                            for test in tests:
                                cursor.execute(f"ALTER TABLE {new_table} ADD {test} INT(3) NOT NULL DEFAULT 0")
                                db.commit()

                            # custom columns to store predictor variables
                            cursor.execute(f"ALTER TABLE {new_table} ADD PASS_CHANCE FLOAT NOT NULL DEFAULT 0")
                            db.commit()
                            cursor.execute(f"ALTER TABLE {new_table} ADD PREDICTED_GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                            db.commit()
                            cursor.execute(f"ALTER TABLE {new_table} ADD GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                            db.commit()

                            # adding each student id for each student record table and this subject
                            cursor.execute(f"SELECT id FROM {table}")
                            for x in [x[0] for x in cursor.fetchall()]:
                                cursor.execute(f"INSERT INTO {new_table} (student_id) VALUES ({x})")
                                db.commit()

                            cursor.execute(
                                f"CREATE TRIGGER {new_table} AFTER INSERT ON {table} FOR EACH ROW INSERT INTO {new_table} (student_id) values (new.id)")
                            db.commit()
                        os.system('cls')
                        print(f"{table} records created\n")
                    else:
                        os.system('cls')
                        print(f"Student record for {year} already exists...\n")

                elif choice == "Add a student account":  # adding students to a record
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
                                        db.commit()

                                        cursor.execute(
                                            f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE        students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {year})")
                                        break
                                    else:
                                        cursor.execute(f"DROP TRIGGER IF EXISTS semester_count_{year}")

                                        cursor.execute(
                                            f"INSERT INTO {student_record} (first_name, last_name, email, username, password, start_sem, entry) VALUES ('{input('Enter first name: ')}', '{input('Enter last name: ')}', '{email}', '{user}', '{input('Enter password: ')}', {sem}, 'Lateral')")
                                        db.commit()

                                        cursor.execute(
                                            f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE        students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {year})")
                                        break

                                cursor.execute(f"SELECT students + lat_students FROM students_batch WHERE start_year = {int(year)}")
                                tot_students = cursor.fetchall()[0][0]
                                cursor.execute(
                                    f"UPDATE students_batch SET tot_students = {tot_students} WHERE start_year = {int(year)}")
                                db.commit()
                                os.system('cls')
                                print(f"{user} has been added as a student for the {year} batch\n")
                                break
                            else:
                                os.system('cls')
                                print(f"{user} already exists, try again\n")
                    else:
                        os.system('cls')
                        print("Going back...\n")
                        continue

                elif choice == "Delete a student account":  # deleting a student from a record
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
                        user = input("Input username of student to delete : ")

                        cursor.execute(f"SELECT username FROM {student_record}")
                        users = [x[0] for x in cursor.fetchall()]
                        if user not in users:
                            print(f"Invalid user: {user}, going back...\n")
                            continue

                        print(f"\nAre you sure to delete {user}?\n")
                        choice = questionary.select("Choices: ", choices=["Yes", "No"]).ask()
                        if choice == "Yes":
                            cursor.execute(f"DROP TRIGGER IF EXISTS semester_count_{year}")

                            cursor.execute(f"DELETE FROM {student_record} WHERE username = '{user}'")
                            db.commit()

                            cursor.execute(
                                f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {year})")

                            cursor.execute(
                                f"SELECT students + lat_students FROM students_batch WHERE start_year = {year}")
                            tot_students = cursor.fetchall()[0][0]
                            cursor.execute(
                                f"UPDATE students_batch SET tot_students = {tot_students} WHERE start_year = {year}")
                            db.commit()

                            os.system('cls')
                            print(f"{user} has been removed from {student_record}\n")
                        else:
                            os.system('cls')
                            print("Going back...\n")
                            continue
                    else:
                        os.system('cls')
                        print("Going back...\n")
                        continue

                elif choice == "Go Back":
                    os.system('cls')
                    print("Going back...\n")
                    break

        elif choice == "Manage my account details":  # checking account details for this admin account
            print("Viewing account details\n")
            cursor.execute("DESC courses.admins")
            cols = [x[0] for x in cursor.fetchall()]

            cursor.execute(f"SELECT * FROM courses.admins WHERE courses.admins.username = '{user}'")
            details = cursor.fetchall()[0]

            # asking whether to view or change account details
            choice = questionary.select("Choice : ", choices=["View account details", "Change account details", "Go Back"]).ask()
            if choice == "View account details":
                [print(f"{cols[x]} -> {details[x]}") for x in range(len(cols))]
                print(f"\nAccount details shown for {user}")
                input("\nPress anything to continue...\n")
                os.system('cls')

            elif choice == "Change account details":
                cursor.execute("DESC courses.admins")

                # getting account details that can be changed
                cols = [x[0] for x in cursor.fetchall() if x[0] not in ['id', 'email']]
                cols.append("Go Back")

                print("Which detail to change?\n")
                col = questionary.select("Choice : ", cols).ask()

                if col != "Go Back":
                    while True:
                        try:
                            cursor.execute(f"SELECT id FROM courses.admins WHERE username = '{user}'")
                            id = cursor.fetchall()[0][0]

                            cursor.execute(f"SELECT {col} FROM courses.admins WHERE username = '{user}'")
                            old_detail = cursor.fetchall()[0][0]
                            new_detail = input(f"\nOld {col} -> {old_detail}\nNew {col} -> ")

                            cursor.execute(f"UPDATE courses.admins SET courses.admins.{col} = '{new_detail}' WHERE courses.admins.username = '{user}'")
                            db.commit()

                            cursor.execute(f"SELECT username FROM courses.admins WHERE id = '{id}'")
                            user = cursor.fetchall()[0][0]
                            os.system('cls')
                            print(f"{col} changed for {user}\n")
                            break
                        except:
                            print("Enter valid input...\n")

                else:
                    os.system('cls')
                    print("Going back...\n")
                    continue

            elif choice == "Go Back":
                os.system('cls')
                print("Going back...\n")
                continue

        elif choice == "Add admin account":  # add an admin account
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
                        f"INSERT INTO courses.admins (first_name, last_name, username, password, email, added_by) VALUES ('{first_name}', '{last_name}', '{username}', '{passw}', '{email}', '{user}')")
                    db.commit()
                    print(f"{username} has been added as an admin\n")
                    os.system('cls')
                    break
                except:
                    print("Enter valid details...\n")

        elif choice == 'Logout':
            os.system('cls')
            print("Logging out...\n")
            break


def admin_auth():
    print("Admin Login\n")
    cursor.execute("SELECT username, password FROM courses.admins")
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
            admin_session(user)

    if valid_login == False:
        os.system('cls')
        print("Invalid admin login details...\n")


def course_select():  # accessing tables in database courses to select a course to access
    cursor.execute("SHOW DATABASES LIKE 'courses'")
    dbs = [x[0] for x in cursor.fetchall()]

    if len(dbs) < 1:  # if no database 'courses' is found, creating it and adding necessary tables with initial values for them
        print("No courses database found, creating one...\n")
        # creating courses database if it doesn't exist
        cursor.execute("CREATE DATABASE courses")
        db.commit()

        cursor.execute("USE courses")
        db.commit()

        # creating table in courses database to store courses
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS courses (id VARCHAR(10) NOT NULL, name TEXT, founded INT, length INT, department TEXT DEFAULT NULL, department_head TEXT DEFAULT NULL, PRIMARY KEY(id))")
        db.commit()

        # adding a course into the table
        print("Courses database created, input details for a course for courses table: \n")
        while True:
            try:
                id = input("Enter course id : ").replace(' ', '_')
                name = input("Full name of course : ")
                founded = int(input("What year was the course founded? : "))
                length = int(input("How many years does the course last? : "))
                department = input("Which department does this course belong to? : ")
                department_head = input("Name of department head : ")

                cursor.execute(f"INSERT INTO courses.courses VALUES ('{id}', '{name}', {founded}, {length}, '{department}', '{department_head}')")
                db.commit()
                print(f"{id}, {name} successfully added\n")
                break
            except:
                print("Enter appropriate values for each field...\n")

        # creating admins table
        cursor.execute("CREATE TABLE IF NOT EXISTS admins (id INT AUTO_INCREMENT PRIMARY KEY, first_name TEXT DEFAULT NULL, last_name TEXT DEFAULT NULL, username VARCHAR(20) UNIQUE NOT NULL, password VARCHAR(20) NOT NULL, email TEXT NOT NULL, mobile VARCHAR(15) DEFAULT NULL, added_by VARCHAR(20) DEFAULT NULL)")
        db.commit()

        os.system('cls')
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
                db.commit()
                print(f"{username} added as an admin")
                break
                input("\nCourse database created, press anything to continue...\n")
                os.system('cls')
            except:
                print("Enter appropriate values for each field...\n")

    # accessing courses database
    cursor.execute("USE courses")
    db.commit()

    # accessing course ids from courses table
    cursor.execute("SELECT id FROM courses")
    courses = [x[0] for x in cursor.fetchall()]
    courses.append("Exit")

    # choosing course
    print("Select course:\n")
    course = questionary.select("Choice : ", choices=courses).ask()

    if course != "Exit":
        cursor.execute(f"SELECT id, name, length FROM courses WHERE id = '{course}'")
        return cursor.fetchall()[0]
    else:
        exit("Bye!")


def prerequisite_tables(database):
    # creating database for course if it doesnt exist already
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
    db.commit()

    cursor.execute(f"USE {database}")
    db.commit()

    # creating prerequisite tables needed for the system to function if they dont exist already
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS teachers (teacher_id INT(5) NOT NULL AUTO_INCREMENT, first_name VARCHAR(30) NOT NULL, last_name VARCHAR(30) NOT NULL, username VARCHAR(20) NOT NULL,email TEXT(50) NULL, password VARCHAR(20) NOT NULL,  PRIMARY KEY (teacher_id), UNIQUE (username))")
    db.commit()

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS subjects (id VARCHAR(10) NOT NULL , name TEXT NOT NULL ,semester INT(2) NOT NULL , teacher_id INT(5) NULL , PRIMARY KEY (id))")
    db.commit()

    # creating foreign key for subjects and teachers for teacher_id
    cursor.execute("SHOW TABLES LIKE 'subjects'")
    if len([x[0] for x in cursor.fetchall()]) < 1:
        cursor.execute(
            "ALTER TABLE subjects ADD FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL ON UPDATE CASCADE")
        db.commit()

    # creating table to store students semester count, year start and expected year of graduation
    cursor.execute("CREATE TABLE IF NOT EXISTS students_batch (start_year INT NOT NULL AUTO_INCREMENT, grad_year INT, students INT NOT NULL DEFAULT 0, lat_students INT DEFAULT 0, tot_students INT DEFAULT 0, cur_semester INT DEFAULT 1, PRIMARY KEY (start_year))")
    db.commit()


def main():
    # running function to select a course (or add prerequisite tables for courses database)

    global database
    global course
    global course_len
    global path

    database, course, course_len = map(str, course_select())
    course_len = int(course_len)

    prerequisite_tables(database)  # checking if prerequisite tables for the database are there, if not, create them

    """ PATH TO SQL DATABASES STORED ON YOUR DEVICE """
    path = f"C:/ProgramData/MySQL/MySQL Server 8.0/Data/{database}"

    while True:
        choice = questionary.select(f"{course} Login Page", choices=["Admin Login",
                                                                     "Teacher Login",
                                                                     "Student Login",
                                                                     "Exit"]).ask()

        if choice == 'Admin Login':
            os.system('cls')
            admin_auth()

        elif choice == 'Teacher Login':
            os.system('cls')
            teacher_auth()

        elif choice == 'Student Login':
            os.system('cls')
            student_auth()

        else:
            exit("Bye!")


# war begins, ionia calls. hasagi
if __name__ == "__main__":
    db = mysql.connect(host='localhost', user='Ashwin', password='3431')  # initializing connection to MySQL
    cursor = db.cursor(buffered=True)
    main()
