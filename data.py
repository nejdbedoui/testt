import pandas as pd
import random

# Define the possible values for each column
genders = ['Male', 'Female']
premium_options = [0, 1]
categories = ['Technology and Programming', 'Science and Mathematics', 'Literature', 'History', 'Art']


# Function to generate a single record with adjusted logic
def generate_record():
    gender = random.choice(genders)
    category = random.choices(categories, weights=[0.3, 0.3, 0.2, 0.1, 0.1], k=1)[0]

    if category == 'Technology and Programming':
        required_lvl = random.randint(30, 50)
        premium = random.choices(premium_options, weights=[0.2, 0.8], k=1)[0]  # Higher chance of premium
    elif category == 'Science and Mathematics':
        required_lvl = random.randint(20, 50)
        premium = random.choices(premium_options, weights=[0.2, 0.8], k=1)[0]  # Higher chance of premium
    elif category == 'Literature':
        required_lvl = random.randint(11, 20)
        premium = random.choices(premium_options, weights=[0.5, 0.5], k=1)[0]  # Equal chance of premium
    elif category == 'History':
        required_lvl = random.randint(5, 15)
        premium = random.choices(premium_options, weights=[0.7, 0.3], k=1)[0]  # Lower chance of premium
    elif category == 'Art':
        required_lvl = random.randint(0, 10)
        premium = random.choices(premium_options, weights=[0.8, 0.2], k=1)[0]  # Lower chance of premium

    age = random.randint(15, 65)  # Assuming typical age range for users
    return {'gender': gender, 'age': age, 'premium': premium, 'required_lvl': required_lvl, 'category': category}


# Generate a list of records
def generate_data(num_records):
    data = [generate_record() for _ in range(num_records)]
    return pd.DataFrame(data)


# Generate 1000 records
df = generate_data(160000)

# Save to CSV
df.to_csv('generated_data.csv', index=False)
print("Data generation complete. File saved as 'generated_data.csv'")
