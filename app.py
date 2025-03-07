# STEP 3 : HOSTING THE ATTENDANCE IN WEB BROWZER

# Note : If Python app.py command does not run then use streamlit run app.py(streamlit modules to be installed before running)

# Import necessary libraries
import streamlit as st  # Streamlit for creating a web application
import pandas as pd  # Pandas for handling and displaying tabular data
import time  # Time module to work with timestamps
from datetime import datetime  # Datetime module to format timestamps

# Get the current timestamp
ts = time.time()

# Format the current date as "day-month-year" (e.g., "07-03-2025")
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

# Format the current time as "hour:minute-seconds" (e.g., "14:30-45")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# Import the Streamlit auto-refresh module to refresh the page at set intervals
from streamlit_autorefresh import st_autorefresh

# Auto-refresh the page every 2000 milliseconds (2 seconds) up to 100 times
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# Display a message based on the current count value
if count == 0:
    st.write("Count is zero")  # If count is zero, display this message
elif count % 3 == 0 and count % 5 == 0:
    st.write("MULTIPLE OF 3 AND 5")  # If count is a multiple of both 3 and 5, display "MULTIPLE OF 3 AND 5"
elif count % 3 == 0:
    st.write("MULTIPLE OF 3")  # If count is a multiple of 3, display "MULTIPLE OF 3"
elif count % 5 == 0:
    st.write("MULTIPLE OF 5")  # If count is a multiple of 5, display "MULTIPLE OF 5"
else:
    st.write(f"Count: {count}")  # Otherwise, display the current count

# Read the attendance CSV file for the current date
df = pd.read_csv("Attendance/Attendance_" + date + ".csv")

# Display the attendance dataframe with highlighted maximum values in each column
st.dataframe(df.style.highlight_max(axis=0))
