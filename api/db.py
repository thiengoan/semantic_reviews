import mysql.connector

# Establish the connection
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="sematic_review"
)

# Create a cursor object
cursor = connection.cursor()

# Execute a query
cursor.execute("SELECT * FROM reviews")

# Fetch the results
results = cursor.fetchall()

for row in results:
    print(row)

# Close the cursor and connection
cursor.close()
connection.close()