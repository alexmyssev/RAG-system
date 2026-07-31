import os

class Authorization:

    def add_new_user(self, users):
        username = input("Enter username: ").strip()
        if username in users:
            raise ValueError("Such username already exists! Please enter another one.")
        password = input("Enter password: ").strip()
        password2 = input("Repeat your password: ").strip()
        if password != password2:
            raise ValueError("Passwords do not match! Please try again.")

        role = "user"

        with open ("users.txt", 'a', encoding = "utf-8") as file:
            file.write(f"{username}:{password}:{role}\n")

        return role

    def read_file(self):
        users = {}
        if not os.path.exists("users.txt"):
            raise ValueError("Users file does not exist!")

        with open ("users.txt", 'r', encoding = "utf-8") as file:
            for line in file:
                st = line.strip()
                if st == "":
                    return users
                username, password, role = st.split(":")
                users[username] = {"password": password, "role": role}
        return users

    def sign_in(self):

        users = self.read_file()

        no_answer = ["no", "n"]
        while True:
            print("Do you want to sign in?")
            answer = input("Y or N: ").strip().lower()
            if answer in no_answer:
                return "user"
            else:
                print("Do you have an account?")
                answer = input("Y or N: ").strip().lower()
                if answer in no_answer:
                    role = self.add_new_user(users)
                    return role
                else:
                    username = input("Enter username: ").strip()
                    password = input("Enter password: ").strip()
                    for user in users:
                        if user == username and users[username]["password"] == password:
                            return users[username]["role"]
                    print("Incorrect username or password! Please try again.")
