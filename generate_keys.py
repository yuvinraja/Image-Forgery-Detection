# from streamlit_authenticator.utilities.hasher import Hasher
import streamlit_authenticator as stauth

# User details
names = ["Yuvin Raja", "Sundar Murthy"]
usernames = ["yuvinraja", "sundarmurthy"]
passwords = ["yuv123!", "sun456#"]


# hashed_passwords = Hasher(passwords).generate()
hashed_passwords = stauth.Hasher(passwords).generate();

print(hashed_passwords)
