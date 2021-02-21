import os
import math
import warnings
import seaborn as sns
import mysql.connector as mysql
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import pickle
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as gs
plt.style.use('bmh')
sns.set_style('dark')
warnings.filterwarnings('ignore')


""" MAKE SURE THE VALUES FOR THE 4 BOTTOM LINES ARE CORRECT FOR YOU """
path = "C:/xampp/mysql/data/btechcse"  # path to database and to store prediction models
database = input("Which database to look at? : ")  # choosing which database to access (btechcse)
db = mysql.connect(host='localhost', user='Ashwin', password='3431', database=database)  # initializing connection to database
cursor = db.cursor(buffered=True)  # cursor


def grades(test_type, test_amount, max_mark, weightage, pass_percent, final_test_name, n=1000, graphs=False):  # grades generator
    """Func that generates train/test data for the classifier/regressor and returns the trained classifer/regressor"""
    df = pd.DataFrame(index=range(1, n+1))  # making the dataframe and generating dummy marks
    df.index.name = 'Student'
    print("\nGenerating mock data\n")
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

    X = df.drop(['Pass/Fail', 'Total %', final_test_name], axis=1)
    y = df[['Pass/Fail', 'Total %']].copy()
    y['Pass/Fail'] = LabelEncoder().fit_transform(y['Pass/Fail'])

    print("Creating and fitting models\n")
    # making train and test data for models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y['Pass/Fail'])

    # passing probability predictor
    passfail = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.arange(0.1, 1.1, 0.1),
                                                                    cv=RepeatedStratifiedKFold(n_splits=10, random_state=7),
                                                                    max_iter=1000, n_jobs=-1, refit=True,
                                                                    random_state=7,
                                                                    class_weight='balanced')).fit(X_test,
                                                                                                  y_test['Pass/Fail'])

    if graphs == True:
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

    # final overall grade predictor
    overallgrade = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1)).fit(X_test, y_test['Total %'])
    print("Models created")

    print("LogisticRegressionCV classifer:-")
    print(f"Accuracy -> {round(passfail.score(X_test, y_test['Pass/Fail'])*100, 2)}%")
    print(f"f1_score -> {round(f1_score(y_test['Pass/Fail'], passfail.predict(X_test))*100, 2)}%\n")

    print("LinearRegression regressor:-")
    print(f"Accuracy -> {round(overallgrade.score(X_test, y_test['Total %'])*100, 2)}%")
    print("Models created")

    return passfail, overallgrade


def rolling_predict(marks, subject, record):  # to present rolling predictions based on a student's marks
    # loading subject prediction models
    passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
    overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

    dummy = [0] * len(marks[0])  # making dummy list to cummulatively add each test score
    pass_probabs = []  # to store each probability as each test score gets entered
    for x in range(len(marks[0])):
        dummy[x] = marks[0][x]
        pass_probabs.append(passfail.predict_proba(np.array([dummy]))[0][1] * 100)

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

    # getting the name of tests used for predictions
    cursor.execute(f"DESCRIBE {record}_{subject}")
    tests = [x[0] for x in cursor.fetchall()][1:-3]

    fig = plt.figure(figsize=(14, 7))
    grid = gs(nrows=1, ncols=2, figure=fig)
    plt.suptitle(
        f'Chance of Passing and Predicted Total Grade for {subject}\nBe warned, these numbers are not supposed to be an extremely accurate representation of your future')

    ax1 = fig.add_subplot(grid[:1])
    plt.title(f"Probability of passing the subject after each test taken\nPredicted Pass or Fail? -> {pf}\
    \nChance of passing subject -> {passfail.predict_proba(marks)[0][1] * 100:.2f}%", fontsize=14)
    plt.plot(tests, pass_probabs, c='black', linestyle='--', label='Predicted passing chance')
    plt.axhline(50, color='r', label="Threshold")
    plt.xticks(rotation=45)
    plt.ylabel('Probability (%)')
    plt.ylim(ymin=50-limit1, ymax=50+limit1)
    plt.margins(0.02, 0.02)
    plt.legend(loc='best')
    plt.tight_layout()

    ax2 = fig.add_subplot(grid[1:])
    plt.title(f"Predicting Overall grade after each test\nPredicted Overall Subject Grade -> {grade_p:.2f}%\
    \nPredicted Grade -> {grade}",
              fontsize=14)
    plt.plot(tests, total_percent, c='black', linestyle='--', label='Predicted Overall grade')
    plt.axhline(60, color='r', label='Passing Threshold')
    plt.xticks(rotation=45)
    plt.ylabel('Final Grade')
    plt.margins(x=0.01, y=0.01)
    plt.ylim(ymin=60-limit2, ymax=60+limit2)
    plt.margins(0.02, 0.02)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    print("\nDetailed predictions for marks shown\n")


def student_session(record, user):
    while True:
        print("\n1. View grades")
        print("2. View account details")
        print("3. Change account details")
        print("4. Logout")
        choice = input("Choice : ")
        if choice == '1':  # view grades of a subject
            print("\nView grades\b")
            cursor.execute("SELECT id, name FROM subjects")
            subjects = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
            [print(f"{x[0]}. {x[1]}") for x in subjects]  # choosing subject
            print(f"{len(subjects)+1}. Go back")
            choice = input("Choice : ")
            if int(choice) in [x[0] for x in subjects]:
                subject = subjects[int(choice)-1][1]
                cursor.execute(f"SELECT id FROM {record} WHERE username = '{user}'")
                id = cursor.fetchall()[0][0]
                cursor.execute(f"DESCRIBE {record}_{subject}")
                tests = [x[0] for x in cursor.fetchall() if x[0] != 'student_id']
                print("")
                for test in tests:  # printing marks on a subject for each test
                    cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {id}")
                    print(f"{test} -> {cursor.fetchall()[0][0]}")

                marks = []  # storing marks of student
                print("\n1. View more info")  # to show prediction result graphs
                print("2. Go back")
                choice = input("Choice : ")
                if choice == '1':
                    cursor.execute(f"DESCRIBE {record}_{subject}")
                    tests = [x[0] for x in cursor.fetchall()][1:-3]
                    for test in tests:  # iterating and getting marks for tests except for the final test
                        cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {id}")
                        marks.append(cursor.fetchall()[0][0])
                    marks = np.array(marks).reshape(1, -1)  # prepping data to be used for predictions
                    rolling_predict(marks, subject, record)
                    print(f"\nMarks displayed for {user} in {subject}\n")
                else:
                    print("\nGoing back\n")
            else:
                print("\nGoing back\n")
                continue

        elif choice == '2':  # showing student account details
            print("\nShowing account details\n")
            cursor.execute(f"DESCRIBE {record}")
            cols = [x[0] for x in cursor.fetchall()]
            for col in cols:
                cursor.execute(f"SELECT {col} FROM {record} WHERE username = '{user}'")
                print(f"{col} -> {cursor.fetchall()[0][0]}")
            print(f"\nAccount details for username {user} shown\n")

        elif choice == '3':  # changing detail in student account
            print(f"\nChanging account details for {user}\n")
            cursor.execute(f"DESCRIBE {record}")
            cols = list(enumerate([x[0] for x in cursor.fetchall() if x[0] not in ["id", "email", "username"]], start=1))
            [print(f"{x[0]}. {x[1]}") for x in cols]
            print(f"{len(cols)+1} Go back")
            choice = input("Choice : ")
            if int(choice) in [x[0] for x in cols]:
                col = cols[int(choice)-1][1]
                cursor.execute(f"SELECT {col} FROM {record} WHERE username = '{user}'")
                old_detail = cursor.fetchall()[0][0]
                new_detail = input(f"\nOld {col} -> {old_detail}\nEnter new {col} : ")
                cursor.execute(f"UPDATE {record} SET {col} = '{new_detail}' WHERE username = '{user}'")
                db.commit()
                print(f"{user} {col} changed from {old_detail} to {new_detail}\n")
            else:
                print("Going back\n")
                continue

        elif choice == '4':
            print("Logging out\n")
            break

        else:
            print("Enter valid choice\n")


def student_auth():
    cursor.execute("SHOW TABLES LIKE 'students%'")
    records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 15], start=1))
    print("\nWhich record do you belong to?")
    [print(f"{x[0]}. {x[1]}") for x in records]  # choosing which student record they belong to
    choice = input("Choice : ")
    if int(choice) in [x[0] for x in records]:
        record = records[int(choice)-1][1]
        print("\nStudent Login\n")
        user = input("Enter username : ")  # checking login details
        passw = input("Enter password : ")
        cursor.execute(f"SELECT username FROM {record}")
        if user in [x[0] for x in cursor.fetchall()]:
            cursor.execute(f"SELECT password FROM {record} WHERE username = '{user}'")
            if passw == cursor.fetchall()[0][0]:
                student_session(record, user)
            else:
                print("\nImcorrect details, going back\n")
        else:
            print("\nIncorrect details, going back\n")
    else:
        print("Going back\n")


def teacher_session(teacher_id):
    cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = {teacher_id}")
    subjects = list(enumerate([x[0] for x in cursor.fetchall()], start=1))  # keeping a track of this particular teacher's subjects
    while True:
        print("\nWelcome to the teacher menu\n")
        print("1. Review overall marks for a subject")
        print("2. Update a test score")
        print("3. Update a student's score")
        print("4. Edit account info")
        print("5. Logout")
        choice = input("Choice : ")
        if choice == '1':  # review marks for a subjects
            print("For which subject?\n")
            [print(f"{x[0]}. {x[1]}") for x in subjects]
            print(f"{len(subjects)+1}. Go back")
            choice = input("Choice : ")
            # choosing for which subject to edit marks
            if int(choice) in [x[0] for x in subjects]:
                print("")
                subject = subjects[int(choice)-1][1]
                cursor.execute("SHOW TABLES LIKE 'students%'")
                # choosing for which batch to edit marks for
                records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 15], start=1))
                print("Choose student batch\n")
                [print(f"{x[0]}. {x[1]}") for x in records]
                print(f"{len(records)+1}. Go back")
                choice = input("Choice : ")
                if int(choice) in [x[0] for x in records]:
                    record = records[int(choice)-1][1]
                    table = f"{record}_{subject}".lower()
                    print(table)
                    print(pd.read_sql(table, con=create_engine(f"mysql://root:@localhost/{database}").connect()))
                    print(f"\n{record}_{subject} values shown\n")
                else:
                    print("Going back\n")
                    continue

            else:
                print("Going back\n")
                continue

        elif choice == '2':  # updating scores of students for a test
            print("For which subject?\n")
            [print(f"{x[0]}. {x[1]}") for x in subjects]
            print(f"{len(subjects)+1}. Go back")
            choice = input("Choice : ")
            # choosing for which subject to edit marks
            if int(choice) in [x[0] for x in subjects]:
                print("")
                subject = subjects[int(choice)-1][1]
                cursor.execute("SHOW TABLES LIKE 'students%'")
                # choosing for which batch to edit marks for
                records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 15], start=1))
                print("Choose student batch\n")
                [print(f"{x[0]}. {x[1]}") for x in records]
                print(f"{len(records)+1}. Go back")
                choice = input("Choice : ")
                if int(choice) in [x[0] for x in records]:
                    record = records[int(choice)-1][1]
                    # choosing which test to edit marks for
                    cursor.execute(f"DESCRIBE {record}_{subject}")
                    tests = list(enumerate([x[0] for x in cursor.fetchall() if x[0] not in ["student_id", "PASS_CHANCE", "PREDICTED_GRADE"]], start=1))
                    print("\nWhich test?\n")
                    [print(f"{x[0]}. {x[1]}") for x in tests]
                    print(f"{len(tests)}. Go back")
                    choice = input("Choice : ")

                    # loading subject prediction models
                    passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
                    overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

                    if int(choice) in [x[0] for x in tests]:
                        test = tests[int(choice)-1][1]
                        print(f"\nEditing marks of {test} for {record} for subject {subject}\n")
                        # getting the list of student_ids in this mark sheet
                        cursor.execute(f"SELECT student_id FROM {record}_{subject}")
                        ids = [x[0] for x in cursor.fetchall()]
                        print("\nEnter marks\n")
                        for id in ids:
                            mark = int(input(f"Student {id} -> "))
                            # update mark of each student
                            cursor.execute(f"UPDATE {record}_{subject} SET {test} = {mark} WHERE student_id = {id}")
                            db.commit()

                        """THERE WAS SOMETHING STRANGE THAT I ENCOUNTERED HERE WHERE BOTH THE THINGS OPERATIONS IN THE 2 "FOR ID IN IDS" LOOPS CANNOT BE DONE UNDER ONE OF THESE LOOPS AND INSTEAD HAD TO BE DONE SEPERATELY UNDER ITS OWN LOOP"""

                        for id in ids:
                            cursor.execute(f"DESCRIBE {record}_{subject}")
                            # get the name of tests for prediction
                            tests = [x[0] for x in cursor.fetchall()][1:-3]
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
                        print("Marks updated\n")

                    else:
                        print("Going back\n")
                        continue

                else:
                    print("Going back\n")
                    continue

            elif int(choice) == len(subjects)+1:
                print("Going back\n")
                continue

        elif choice == '3':  # updating grade of one studnet in a particular subject
            print("For which subject?\n")
            [print(f"{x[0]}. {x[1]}") for x in subjects]
            print(f"{len(subjects)+1}. Go back")
            choice = input("Choice : ")
            # choosing for which subject to edit marks
            if int(choice) in [x[0] for x in subjects]:
                print("")
                subject = subjects[int(choice)-1][1]
                cursor.execute("SHOW TABLES LIKE 'students%'")
                # choosing for which batch to edit marks for
                records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 15], start=1))
                print("Choose student batch\n")
                [print(f"{x[0]}. {x[1]}") for x in records]
                print(f"{len(records)+1}. Go back")
                choice = input("Choice : ")
                if int(choice) in [x[0] for x in records]:
                    record = records[int(choice)-1][1]
                    print("Choose student id to edit marks for : \n")
                    cursor.execute(f"SELECT student_id FROM {record}_{subject}")
                    ids = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in ids]
                    print(f"{len(ids)}. Go back")
                    choice = input("Choice : ")
                    if int(choice) in [x[0] for x in ids]:
                        student_id = ids[int(choice)-1][1]
                        cursor.execute(f"DESCRIBE {record}_{subject}")
                        tests = list(enumerate([x[0] for x in cursor.fetchall() if x[0] not in ["student_id", "PASS_CHANCE", "PREDICTED_GRADE"]], start=1))
                        print("\nWhich test?\n")
                        [print(f"{x[0]}. {x[1]}") for x in tests]
                        print(f"{len(tests)}. Go back")
                        choice = input("Choice : ")
                        if int(choice) in [x[0] for x in tests]:
                            test = tests[int(choice)-1][1]
                            mark = input(f"\nEnter new mark for student_id {student_id} in {test} for {subject}\n")
                            cursor.execute(f"UPDATE {record}_{subject} SET {test} = {int(mark)} WHERE student_id = {student_id}")
                            db.commit()
                            print(f"Mark updated for student_id {student_id} in {test} for {subject}\n")
                    else:
                        print("Going back\n")
                else:
                    print("Going back\n")
            else:
                print("Going back\n")

        elif choice == '4':  # changing first_name, last_name or password of teacher account
            print("\nChanging account detail\n")
            cursor.execute("DESCRIBE teachers")
            details = list(enumerate([x[0] for x in cursor.fetchall() if x[0] not in ['teacher_id', 'username']], start=1))
            [print(f"{x[0]}. {x[1]}") for x in details]
            print(f"{len(details)+1} Go back")
            choice = input("Choice : ")
            if int(choice) in [x[0] for x in details]:
                detail = details[int(choice)-1][1]
                print(f"\nChanging {detail}\n")
                new_detail = input(f"Enter new {detail} : ")
                cursor.execute(f"UPDATE teachers SET {detail} = '{new_detail}' WHERE teacher_id = {teacher_id}")
                db.commit()
                print(f"{detail} updated\n")

            else:
                print("Going back")
                continue

        elif choice == '5':
            print("Logging out\n")
            break

        else:
            print("No valid option entered\n")


def teacher_auth():
    user = input("\nEnter username : ")
    passw = input("Enter password : ")
    cursor.execute("SELECT username FROM teachers")
    if user in [x[0] for x in cursor.fetchall()]:
        cursor.execute(f"SELECT password FROM teachers WHERE username = '{user}'")
        if passw == cursor.fetchall()[0][0]:
            cursor.execute(f"SELECT teacher_id FROM teachers WHERE username = '{user}'")
            teacher_session(cursor.fetchall()[0][0])
        else:
            print("\nWrong password\n")
    else:
        print("\nWrong username\n")


def admin_session():
    while True:
        print("\nWelcome to the admin menu\n")
        print("1. Set up main tables (students_{year} , teachers and subjects)")
        print("2. Show tables")
        print("3. Manage subjects")
        print("4. Manage teacher accounts")
        print("5. Manage student account")
        print("6. Logout")
        choice = input("Choice : ")

        if choice == '1':  # automatically setting up the main 3 prerequisite tables required for further operations in course
            cursor.execute("SHOW TABLES LIKE 'subjects'")
            if len(cursor.fetchall()) < 1:
                year = input("Enter what year to make a students table for : ")

                print("\nCreating teacher table")
                cursor.execute(
                    f"CREATE TABLE {database}.teachers (teacher_id INT(5) NOT NULL AUTO_INCREMENT , first_name VARCHAR(30) NOT NULL , last_name VARCHAR(30) NOT NULL , username VARCHAR(20) NOT NULL , password VARCHAR(20) NOT NULL , PRIMARY KEY (teacher_id), UNIQUE (username))")
                db.commit()

                print("\nCreating subjects table")
                cursor.execute(
                    f"CREATE TABLE {database}.subjects (id VARCHAR(10) NOT NULL , name TEXT NOT NULL ,semester INT(2) NOT NULL , teacher_id INT(5) NULL , PRIMARY KEY (`id`), INDEX (`teacher_id`)) ENGINE = InnoDB")
                db.commit()

                # creating foreign key for subjects and teachers for teacher_id
                cursor.execute(
                    "ALTER TABLE subjects ADD FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL ON UPDATE CASCADE")
                db.commit()

                print("\nCreating students table")
                cursor.execute(
                    f"CREATE TABLE {database}.students_{year}(id INT(5) NOT NULL AUTO_INCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, mobile_no VARCHAR(15) DEFAULT NULL, email VARCHAR(40) DEFAULT NULL, username VARCHAR(20) NOT NULL, password VARCHAR(20) NOT NULL, PRIMARY KEY(id), UNIQUE (username))")
                db.commit()
                print(f"\nstudents_{year}, teachers and subjects table successfully created\n")

            else:
                print("Main tables seem to already have been created\n")

        elif choice == '2':  # showing all table names and details about them
            cursor.execute("SHOW TABLES")
            tables = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
            print('')
            while True:
                [print(f"{x[0]}. {x[1]}") for x in tables]
                print(f"{len(tables)+1}. Go back")
                choice = input("Option : ")
                if int(choice) in range(1, len(tables)+1):
                    table = tables[int(choice)-1][1]
                    cursor.execute(f"DESCRIBE {table}")
                    print('\n')
                    print(pd.DataFrame(cursor.fetchall()), '\n')

                    print(pd.read_sql(table, con=create_engine(f"mysql://root:@localhost/{database}").connect(), index_col=1))
                    print(f"\n{table} details shown\n")

                elif int(choice) == len(tables)+1:
                    print("\nGoing back\n")
                    break

                else:
                    print("Enter valid choice\n")

        elif choice == '3':  # adding or deleting a subject
            while True:
                print("\n1. Add a subject")
                print("2. Delete a subject")
                print("3. Go back")
                choice = input("Choice : ")
                if choice == '1':
                    print("\nAdding a subject\n")
                    print("IS THE TEACHER THAT TEACHES THIS SUBJECT IN THE TABLE?\n")
                    print(pd.read_sql('teachers', con=create_engine(f"mysql://root:@localhost/{database}").connect()))
                    print("1. YES")
                    print("2. NO")
                    choice = input("Choice : ")
                    if choice == '1':
                        cursor.execute("SELECT id FROM subjects")
                        subject_names = [x[0] for x in cursor.fetchall()]
                        subj_name = input("Enter abbreviation of subject : ").strip()

                        if subj_name not in subject_names:
                            full_name = input("Enter full name of subject : ").strip()
                            semester = int(input("Which semester is this subject in : "))
                            teach = int(input("Enter teacher ID for this subject : "))
                            cursor.execute(f"INSERT INTO subjects VALUES ('{subj_name}', '{full_name}', {semester}, {teach})")
                            db.commit()

                            table_name = f"{subj_name}_details"
                            # type of evaluations
                            test_type = tuple(input("\nEnter the types of tests (seperated by a space): ").split())
                            print(" ")
                            test_amount = tuple(int(input(f"How many tests for {x}?: "))
                                                for x in test_type)  # amount of tests per evaluation
                            print(" ")
                            max_mark = tuple(int(input(f"{x} out of how many marks?: "))
                                             for x in test_type)  # maximum marks for each type of tests
                            print(" ")

                            while True:
                                weightage = tuple(float(input(f"What is the weightage for {x}?: "))/100 for x in test_type)
                                if np.sum(weightage) == 1.0:
                                    print(" ")
                                    break
                                else:
                                    print("Make sure the weightage for all tests add up to 1.0!\n")
                            pass_percent = float(input("What is the passing percentage threshold?: "))/100

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

                            # getting all student record tables for all years
                            cursor.execute("SHOW TABLES LIKE 'students_%'")
                            tables = [x[0] for x in cursor.fetchall() if len(x[0]) < 15]
                            if len(tables) > 0:
                                # making marking sheets for subjects for all students who have a student record
                                for table in tables:
                                    new_table = f"{table}_{subj_name}"
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

                                    # adding each student id for each student record table and this subject
                                    cursor.execute(f"SELECT id FROM {table}")
                                    for x in [x[0] for x in cursor.fetchall()]:
                                        cursor.execute(f"INSERT INTO {new_table} (student_id) VALUES ({x})")
                                        db.commit()

                                    cursor.execute(
                                        f"CREATE TRIGGER {new_table} AFTER INSERT ON {table} FOR EACH ROW INSERT INTO {new_table} (student_id) values (new.id)")
                                    db.commit()

                                print(f"Grades sheets for {subj_name} created\n")
                            else:
                                print("No student record tables found\n")

                        else:
                            print(f"{subj_name} already exists in the subjects table\n")
                    else:
                        print("Make sure to enter the teacher details in the teacher table first\n")

                elif choice == '2':  # deleting a subject
                    print("\nDeleting a subject\n")
                    cursor.execute("SELECT id FROM subjects")
                    names = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in names]
                    print(f"{len(names)+1}. Go back")
                    choice = input("Choice : ")
                    if int(choice) in range(1, len(names)+1):
                        subject = names[int(choice)-1][1]
                        print(f"Deleting {subject}\n")
                        print("Are you sure?")
                        print("1. YES")
                        print("2. NO")
                        choice = input("Choice : ")
                        if choice == '1':
                            cursor.execute("DELETE FROM subjects WHERE id = %s", (subject,))
                            db.commit()

                            cursor.execute(f"SHOW TABLES LIKE '%{subject}%'")
                            tables = [x[0] for x in cursor.fetchall()]
                            if len(tables) > 0:
                                for table in tables:
                                    cursor.execute(f"DROP TABLE {table}")
                                    db.commit()

                            cursor.execute(f"SHOW TABLES LIKE 'students%'")
                            records = [x[0] for x in cursor.fetchall() if len(x[0]) < 15]
                            for record in records:
                                cursor.execute(f"DROP TRIGGER IF EXISTS {record}_{subject}")
                                db.commit()

                            if os.path.exists(f"{path}/{subject}_passfail"):
                                os.remove(f"{path}/{subject}_passfail")

                            if os.path.exists(f"{path}/{subject}_overallgrade"):
                                os.remove(f"{path}/{subject}_overallgrade")

                            print(f"{subject} has been deleted\n")

                        else:
                            print("Going back\n")
                            break

                elif choice == '3':
                    print("Going back\n")
                    break

                else:
                    print("No valid choice entered\n")

        elif choice == '4':  # managing teacher accounts
            while True:
                print("\n1. Add a teacher account")
                print("2. Delete a teacher account")
                print("3. Go back")
                choice = input("Choice : ")
                if choice == '1':  # deleting a teacher account
                    print("Adding a teacher\n")
                    first = input("First name : ")
                    last = input("Last name : ")
                    user = input("Enter username : ")
                    passw = input("Enter password : ")
                    cursor.execute("INSERT INTO teachers (first_name, last_name, username, password) VALUES (%s, %s, %s, %s)",
                                   (first, last, user, passw))
                    db.commit()
                    print(f"{first} has been added as a teacher\n")

                elif choice == '2':
                    print("Deleting a teacher\n")
                    cursor.execute("SELECT teacher_id FROM teachers")
                    ids = [x[0] for x in cursor.fetchall()]
                    print(pd.read_sql('teachers', con=create_engine(f"mysql://root:@localhost/{database}").connect()))
                    choice = input("Input id of teacher to delete : ")
                    if int(choice) in ids:
                        print("Are you sure?")
                        print("1. Yes")
                        print("2. No")
                        choice = input("Choice : ")
                        if choice == '1':
                            cursor.execute(f"DELETE FROM teachers WHERE teacher_id = {choice}")
                            db.commit()
                            print("Teacher has been deleted\n")
                    else:
                        print("Invalid option entered\n")

                elif choice == '3':
                    print("Going back\n")
                    break

                else:
                    print("NO valid option entered\n")

        elif choice == '5':
            while True:
                print("\n1. Create new student records for new batch")
                print("2. Add a student account")
                print("3. Delete a student account")
                print("4. Go back")
                choice = input("Choice : ")
                if choice == '1':  # creating a new student records table for new batch and grade sheets
                    print("\nCreating new student record\n")
                    year = input("Enter year to create student record for : ")
                    cursor.execute(f"SHOW TABLES LIKE 'students_{year}'")
                    if len([x[0] for x in cursor.fetchall()]) < 1:
                        cursor.execute(
                            f"CREATE TABLE btechcse.students_{year}(id INT(5) NOT NULL AUTO_INCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, mobile_no VARCHAR(15) DEFAULT NULL, email VARCHAR(40) DEFAULT NULL, username VARCHAR(20) NOT NULL, password VARCHAR(20) NOT NULL, PRIMARY KEY(id), UNIQUE(username))")
                        db.commit()

                        table = f"students_{year}"  # name of new student records
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

                            # adding each student id for each student record table and this subject
                            cursor.execute(f"SELECT id FROM {table}")
                            for x in [x[0] for x in cursor.fetchall()]:
                                cursor.execute(f"INSERT INTO {new_table} (student_id) VALUES ({x})")
                                db.commit()

                            cursor.execute(
                                f"CREATE TRIGGER {new_table} AFTER INSERT ON {table} FOR EACH ROW INSERT INTO {new_table} (student_id) values (new.id)")
                            db.commit()
                        print(f"{table} records created\n")

                elif choice == '2':
                    print("\nAdding a student to a record\n")
                    cursor.execute("SHOW TABLES LIKE 'students%'")
                    student_records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 14], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in student_records]
                    print(f"{len(student_records)+1}. Go back")
                    choice = input("Choice : ")
                    if int(choice) in range(1, len(student_records)+1):
                        student_record = student_records[int(choice)-1][1]
                        cursor.execute(f"SELECT username FROM {student_record}")
                        existing_users = [x[0] for x in cursor.fetchall()]
                        year = student_record[len(student_record)-4:]
                        while True:
                            user = input("\nEnter username for new student : ")
                            if user not in existing_users:
                                email = f"{user}_{year}@dypiu.ac.in"
                                cursor.execute(
                                    f"INSERT INTO {student_record} (first_name, last_name, email, username, password) VALUES ('{input('Enter first name: ')}', '{input('Enter last name: ')}', '{email}', '{user}', '{input('Enter password: ')}')")
                                db.commit()
                                print(f"\n{user} has been added as a student\n")
                                break
                            else:
                                print(f"\n{username} already exists, try again\n")
                    else:
                        print('\nGoing back\n')
                        continue

                elif choice == '3':
                    print("\nDeleting a student from a record\n")
                    cursor.execute("SHOW TABLES LIKE 'students%'")
                    student_records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 14], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in student_records]
                    print(f"{len(student_records)+1}. Go back")
                    choice = input("Choice : ")
                    if int(choice) in range(1, len(student_records)+1):
                        student_record = student_records[int(choice)-1][1]
                        user = input("Input username of student to delete : ")
                        print(f"\nAre you sure to delete {user}?")
                        print("1. YES")
                        print("2. NO")
                        choice = input("Choice : ")
                        if choice == '1':
                            cursor.execute(f"DELETE FROM {student_record} WHERE username = '{user}'")
                            db.commit()
                            print(f"\n{user} has been removed from {student_record}\n")

                        else:
                            print("\nGoing back\n")
                            continue

                else:
                    print("\nGoing back\n")
                    break

        elif choice == '6':
            print("Logging out\n")
            break

        else:
            print("Enter valid option\n")


def admin_auth():
    if input("\nEnter username : ") == 'admin' and input("Enter password : ") == '1234':
        admin_session()
    else:
        print("Invalid admin login details\n")


def main():
    while True:
        print("\nMain Menu\n")
        print("1. Log in as admin")
        print("2. Log in as student")
        print("3. Log in as teacher")
        print("4. Exit")
        choice = input("Choice : ")

        if choice == '1':
            admin_auth()

        elif choice == '2':
            student_auth()

        elif choice == '3':
            teacher_auth()

        elif choice == '4':
            print("Bye!")
            break


# war begins
main()
