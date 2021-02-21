import mysql.connector as mysql
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import pickle
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import make_pipeline

database = input("Which database to look at? : ")  # choosing which database to access
db = mysql.connect(host='localhost', user='Ashwin', password='3431', database=database)  # initializing
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

    X = df.drop(['Pass/Fail', 'Total %', 'final'], axis=1)
    y = df[['Pass/Fail', 'Total %']].copy()
    y['Pass/Fail'] = LabelEncoder().fit_transform(y['Pass/Fail'])

    print("Creating and fitting models\n")
    # making train and test data for models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y['Pass/Fail'])

    # passing probability predictor
    passfail = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.arange(0.1, 0.6, 0.1),
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

def student_auth():


def teacher_updategrade(subject, test):  # teacher grading for a subject panel
    print(f"\nWelcome to the menu to grade {test} for {subject}\n")
    cursor.execute(f"SELECT * FROM {subject}")
    n = len(cursor.fetchall())
    for n in range(1, n+1):
        score = int(input(f"Student {n} score -> "))
        cursor.execute(f"UPDATE {subject} SET {test} = '{score}' WHERE student_id = {n}")
        db.commit()
    print(f"Grades for {test} in {subject} updated\n")


def teacher_session(user):  # teacher panel
    while True:
        print(f"\nHello {user}, welcome to the teacher menu\n")
        print("1. Review grades")
        print("2. Update test grades")
        print("3. Update a students grades")
        print("4. Change account password")
        print("5. Logout")
        choice = input("Option : ")
        if choice == '1':  # look at the grades of students in a subject the teacher manages
            print("Review grades for which subject?\n")
            # getting the subjects that this teacher manages
            cursor.execute("SELECT subjects FROM teachers WHERE username = %s", (user,))
            # dynamically creating choice list incase a teacher has more than 1 subject
            subjects = list(enumerate(cursor.fetchall()[0][0].split(), start=1))
            for x, y in subjects:
                print(f"{x}. {y}")
            sub_choice = int(input("Option : "))
            cursor.execute("SHOW tables LIKE %s", (subjects[sub_choice - 1][1],))
            if len(cursor.fetchall()) > 0:
                print(pd.read_sql(subjects[sub_choice - 1][1],
                                  con=create_engine(f"mysql://root:@localhost/{database}").connect()))
                print(f"Grades for {subjects[sub_choice - 1][1]} shown\n")
            else:
                print(f"Grade table for {subjects[sub_choice - 1][1]} does not exist\n")

        elif choice == '2':  # updating test grades for a particular test in a subject
            cursor.execute("SELECT subjects from teachers WHERE username = %s", (user,))
            subjects = cursor.fetchall()[0][0].split()
            subjects = list(enumerate(subjects, start=1))
            while True:
                print('')
                for x in subjects:
                    print(f"{x[0]}. {x[1]}")
                print(f"{len(subjects)+1}. Back to teacher menu")
                choice = input("Choose subject : ")
                if int(choice) in [x[0] for x in subjects]:
                    cursor.execute("SHOW tables LIKE %s", (subjects[int(choice)-1][1], ))

                    if len(cursor.fetchall()) < 1:  # incase if valid choice for subject is made but table for it does not exist
                        print(f"Table for {subjects[int(choice)-1][1]} does not exist\n")
                    else:
                        subject = subjects[int(choice)-1][1]
                        cursor.execute(f"SHOW COLUMNS FROM {subjects[int(choice)-1][1]}")
                        tests = list(enumerate([x[0] for x in cursor.fetchall() if x[0] != 'student_id'], start=1))
                        print(f"\nUpdate grades for which test in {subjects[int(choice)-1][1]}\n")
                        [print(f"{x[0]}. {x[1]}") for x in tests]
                        print(f"{len(tests)+1}. Go back to previous menu")
                        choice = input("Option : ")
                        if int(choice) in range(1, len(tests)+1):  # changing grade of students in a subject for a particular test
                            teacher_updategrade(subject, tests[int(choice)-1][1])

                        elif int(choice) == len(tests)+1:
                            print("Going back to previous menu\n")
                            break

                        else:
                            print("No valid option entered\n")

                elif int(choice) == len(subjects)+1:
                    print("Going back to the teacher menu\n")
                    break

                else:
                    print("Invalid choice\n")

        elif choice == '3':  # updating the marks of a student in a test for a subject
            cursor.execute(f"SELECT subjects FROM teachers WHERE username = '{user}'")
            subjects = list(enumerate(cursor.fetchall()[0][0].split(), start=1))
            [print(f"{x[0]}. {x[1]}") for x in subjects]
            print(f"{len(subjects)+1}. Go back to previous menu")
            choice = input("Option : ")
            if int(choice) in [x[0] for x in subjects]:
                subject = subjects[int(choice)-1][1]
                cursor.execute(f"SELECT * FROM {subject}")
                n = len(cursor.fetchall())
                student_id = int(input("\nID of student whose grade to update : "))
                if student_id in range(1, n+1):
                    cursor.execute(f"SHOW COLUMNS FROM {subject}")
                    tests = list(enumerate([x[0] for x in cursor.fetchall() if x[0] != 'student_id'], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in tests]
                    print(f"{len(tests)+1}. Go back to previous menu")
                    choice = input("Option : ")
                    if int(choice) in range(1, len(tests)+1):
                        test = tests[int(choice)-1][1]
                        marks = int(input(f"\nMarks for student {student_id} : "))
                        cursor.execute(f"UPDATE {subject} SET {test} = {marks} WHERE student_id = {student_id}")
                        db.commit()
                        print(f"{test} marks for student {student_id} in {subject} updated\n")
                    else:
                        print("Invalid input for test choice\n")
                else:
                    print("Invalid input for student_id\n")
            else:
                print("Invalid input for subject choice\n")

        elif choice == '4':
            print(f"\nChanging password for {user}\n")
            cursor.execute(f"SELECT password FROM teachers WHERE username = '{user}'")
            old_pass = cursor.fetchall()[0][0]
            inp_old_pass = input("Enter old password : ")
            if inp_old_pass == old_pass:
                print("Identity verified\n")
                inp_new_pass = input("Enter new password : ")
                cursor.execute(f"UPDATE teachers SET password = '{inp_new_pass}' WHERE username = '{user}'")
                db.commit()
                print("Password changed\n")
            else:
                print("Incorrect password entered\n")

        elif choice == '5':
            print("Logging out of teacher account\n")
            break

        else:
            print("Enter valid choice\n")


def teacher_auth():  # checking if valid teacher login
    print("\nTeacher Login\n")
    user, passw = input("Enter username : ").strip(), input("Enter password : ").strip()
    cursor.execute("SELECT * from teachers WHERE username = %s and password = %s", (user, passw))
    if cursor.rowcount == 0:
        print("Login not recognized\n")
    else:
        teacher_session(user)


def admin_session():  # admin panel
    print("\nAdmin menu\n")
    while True:

        print("1. Register/Edit account")
        print("2. Delete existing account")
        print("3. Show details in table")
        print("4. Add a subject")
        print("5. Delete a subject")
        print("6. Logout")
        choice = input("Option : ")
        if choice == '1':  # registering new account
            while True:
                print("\nRegister/Edit account\n")
                print("1. Register new TEACHER")
                print("2. Register new STUDENT")
                print("3. Edit existing TEACHER account details")
                print("4. Edit existing STUDENT account details")
                print("5. Go back to previous menu")
                choice = input("Choice : ")

                if choice == '1':  # registering new teacher
                    print("\nRegister new TEACHER account\n")
                    user = input("Enter username : ")
                    passw = input("Enter password : ")
                    subjs = input("Enter subjects (space seperated) : ")
                    cursor.execute("INSERT INTO teachers (username, password, subjects) VALUES (%s, %s, %s)", (user, passw, subjs))
                    db.commit()
                    print(f"{user} has been registed as a teacher\n")

                elif choice == '2':  # registering new student
                    print("\nRegister new STUDENT account\n")
                    user = input("Enter username : ")
                    passw = input("Enter password : ")
                    cursor.execute("INSERT INTO students_2019 (username, password) VALUES (%s, %s)", (user, passw))
                    db.commit()
                    print(f"{user} has been registered as a student\n")

                elif choice == '3':  # editing existing teacher account details
                    print("\nEditing existing TEACHER account\n")
                    print(pd.read_sql('teachers', con=create_engine(f"mysql://root:@localhost/{database}").connect()))

                    user = input("Enter username : ")
                    edit = input("Enter which detail to edit : ")
                    query = f"SELECT {edit} FROM teachers WHERE username = '{user}'"
                    cursor.execute(query)
                    before_change = cursor.fetchall()[0][0]

                    change = input("Enter new detail : ")
                    query = f"UPDATE teachers SET {edit} = '{change}' WHERE username = '{user}'"
                    cursor.execute(query)
                    db.commit()
                    print(f"Before -> {before_change}\nNow -> {change}\n")

                elif choice == '4':  # editing existing student account details
                    print("\nEditing existing STUDENT accoutn\n")
                    print(pd.read_sql('students_2019', con=create_engine(f"mysql://root:@localhost/{database}").connect()))
                    user = input("Enter username : ")
                    edit = input("Enter which detail to edit : ")
                    query = f"SELECT {edit} FROM students WHERE username = '{user}'"
                    cursor.execute(query)
                    before_change = cursor.fetchall()[0][0]

                    change = input("Enter new detail : ")
                    query = f"UPDATE students_2019 SET {edit} = '{change}' WHERE username = '{user}'"
                    cursor.execute(query)
                    db.commit()
                    print(f"Before -> {before_change}\nNow -> {change}\n")

                elif choice == '5':
                    print("Going to previous menu\n")
                    break

                else:
                    print("No valid input entered\n")

        elif choice == '2':  # deleting existing account
            print("\nDeleting exiting account")
            user = input("Enter username : ")
            cursor.execute("DELETE FROM users WHERE username = %s", (user,))
            if cursor.rowcount == 0:
                print(f"{user} not found\n")
            else:
                db.commit()
                print(f"{user} has been deleted\n")

        elif choice == '3':  # showing records in table
            print("\nShowing records in table\n")
            cursor.execute("""SHOW tables""")
            tables = list(enumerate(cursor.fetchall(), start=1))
            print("TABLES\n")
            for table in tables:
                print(f"{table[0]}. {table[1][0]}")
            choice = input("Option : ")
            if int(choice) in [x[0] for x in tables]:
                # showing pandas view of sql table
                print(pd.read_sql(tables[int(choice)-1][1][0],
                                  con=create_engine(f"mysql://root:@localhost/{database}").connect()))
                print(f"Records in {tables[int(choice)-1][1][0]} shown\n")
            else:
                print("No valid option entered\n")

        elif choice == '4':  # dynamically creating grade sheet for a subject
            print("\nAdding a subject\n")

            # gathering subject details
            subj_name = input("Name of subject : ")
            cursor.execute("SHOW tables")
            if subj_name not in [x[0] for x in cursor.fetchall()]:
                test_type = tuple(input("Types of tests : ").split())
                if len(test_type) < 1:
                    print("Enter valid test types\n")
                    continue

                test_amount = tuple(int(input(f"How many evaluations for {x}? : ")) for x in test_type)
                if len(test_amount) != len(test_type):
                    print("Enter valid corresponding test amounts\n")
                    continue

                test_marks = tuple(int(input(f"How many marks is {x} out of? : ")) for x in test_type)
                if len(test_marks) != len(test_type):
                    print("Enter valid corresponding test marks\n")
                    continue

                while True:
                    weightage = tuple(float(input(f"What is the weightage for {x}?: "))/100 for x in test_type)
                    if np.sum(weightage) == 1.0 and len(weightage) == len(test_type):
                        break
                    else:
                        print("Make sure the weightage for all tests add up to 100 and are corresponding to the test type\n")

                pass_percent = int(input("What is the overall subject passing percent? : "))/100

                while True:
                    final_test_name = input(f"Which of these tests {test_type} is the final test?: ")
                    if final_test_name in test_type:
                        break
                    else:
                        print("Make sure the name of the final test is enterred correctly!\n")

                # dynamically creating sql table
                df = pd.DataFrame(index=range(1, int(input("How many students? : "))+1))
                df.index.name = 'student_id'
                for x in range(len(test_type)):
                    if test_amount[x] > 1:
                        for y in range(1, test_amount[x]+1):
                            df[f"{test_type[x]}_{y}"] = [0] * len(df)
                    else:
                        df[test_type[x]] = [0] * len(df)
                df.to_sql(con=create_engine(
                    f"mysql://root:@localhost/{database}").connect(), if_exists='replace', name=subj_name)

                passfail, overallgrade = grades(test_type, test_amount, test_marks, weightage, pass_percent, final_test_name)
                with open(f"D:/Python/personal-projects/Grades_prediction/passfail_{subj_name}", 'wb') as f:
                    pickle.dump(passfail, f)
                with open(f"D:/Python/personal-projects/Grades_prediction/overallgrade_{subj_name}", 'wb') as f:
                    pickle.dump(overallgrade, f)

                print(f"Grade sheet for {subj_name} created\n ")
            else:
                print(f"Table for {subj_name} already exists\n")

        elif choice == '5':  # drop a table
            print("\nDelete subject\n")
            table = input("Table to delete : ")
            cursor.execute("DROP TABLE " + table)
            db.commit()
            print(f"{table} has been deleted\n")

        elif choice == '6':  # logging out
            print("Logging out\n")
            break
        else:
            print("No valid input entered\n")


def admin_auth():  # checking if valid admin login
    print("\nAdmin Login\n")
    username, password = input("Enter username : ").strip(), input("Enter password : ").strip()
    if username == 'Ashwin' and password == '3431':
        admin_session()
    else:
        print("Invalid login details")


def main():
    while True:
        # main opening screen before login
        print("Welcome to the college system\n")
        print("1. Login as student")
        print("2. Login as teacher")
        print("3. Login as admin")
        print("4. Exit")

        choice = input("Option : ")
        if choice == '1':
            student_auth()
        elif choice == '2':
            teacher_auth()
        elif choice == '3':
            admin_auth()
        elif choice == '4':
            print("Bye!")
            break
        else:
            print("Valid input has not been entered\n")


# war begins
main()
