import os
import hashlib

class Authorization:

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _check_password(self, password: str, password_hash: str) -> bool:
        return hashlib.sha256(password.encode("utf-8")).hexdigest() == password_hash

    def read_file(self) -> dict:
        users = {}
        if not os.path.exists("users.txt"):
            raise ValueError("Users file does not exist!")

        with open ("users.txt", 'r', encoding = "utf-8") as file:
            for line in file:
                st = line.strip()
                if st == "":
                    return users
                username, password_hash, role = st.split(":")
                users[username] = {"password_hash": password_hash, "role": role}
        return users

    def FA_register(self, username: str, password: str) -> str:
        users = self.read_file()
        if username in users:
            raise ValueError("Such username already exists!")

        role = "user"
        password_hash = self._hash_password(password)
        with open("users.txt", 'a', encoding="utf-8") as file:
            file.write(f"{username}:{password_hash}:{role}\n")

        return role

    def FA_verify(self, username: str, password: str) -> str | None:
        users = self.read_file()
        user = users.get(username)
        if user and self._check_password(password, user["password_hash"]):
            return user["role"]
        return None

    def add_new_user(self, users) -> str:
        username = input("Enter username: ").strip()
        if username in users:
            raise ValueError("Such username already exists! Please enter another one.")
        password = input("Enter password: ").strip()
        password2 = input("Repeat your password: ").strip()
        if password != password2:
            raise ValueError("Passwords do not match! Please try again.")
        password_hash = self._hash_password(password)

        role = "user"

        with open ("users.txt", 'a', encoding = "utf-8") as file:
            file.write(f"{username}:{password_hash}:{role}\n")

        return role


    def sign_in(self) -> str:

        users = self.read_file()
        yes_answer = ["yes", "ye", "y"]
        no_answer = ["no", "n"]
        while True:
            print("Do you want to sign in?")
            answer = input("Y or N: ").strip().lower()
            if answer in no_answer:
                return "user"
            elif(answer in yes_answer):
                print("Do you have an account?")
                answer = input("Y or N: ").strip().lower()
                if answer in no_answer:
                    role = self.add_new_user(users)
                    return role
                else:
                    username = input("Enter username: ").strip()
                    password = input("Enter password: ").strip()
                    for user in users:
                        if user == username and self._check_password(password, users[username]["password_hash"]):
                            return users[username]["role"]
                    print("Incorrect username or password! Please try again.")
            else:
                print("No such option")
                continue
