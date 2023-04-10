import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset into a pandas DataFrame
df = pd.read_csv('data/data_science_jobs.csv')

# Clean the data by removing duplicates, null values, and irrelevant columns
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df = df[['title', 'location', 'description']]

# Define a function to extract key requirements from the job descriptions using regular expressions
def extract_requirements(description):
    # Define regular expressions to match programming languages, machine learning algorithms, and tools
    prog_lang_regex = re.compile(r'python|java|c\+\+|c#|javascript|php|ruby|perl|sql')
    ml_algo_regex = re.compile(r'machine learning|artificial intelligence|deep learning|neural networks|nlp')
    tools_regex = re.compile(r'hadoop|spark|tensorflow|keras|scikit-learn|pytorch|aws|azure|gcp')
    
    # Extract requirements using regular expressions and join them into a string
    requirements = []
    requirements.extend(prog_lang_regex.findall(description.lower()))
    requirements.extend(ml_algo_regex.findall(description.lower()))
    requirements.extend(tools_regex.findall(description.lower()))
    return ', '.join(requirements)

# Apply the extract_requirements function to the description column to create a new column named 'requirements'
df['requirements'] = df['description'].apply(extract_requirements)

# Compute statistics such as the most common requirements, job titles, and locations using pandas
top_requirements = df['requirements'].str.split(', ').explode().value_counts().head(10)
top_titles = df['title'].value_counts().head(10)
top_locations = df['location'].value_counts().head(10)

# Visualize the data using matplotlib to create graphs and charts that illustrate the key findings
plt.barh(top_requirements.index, top_requirements.values)
plt.title('Top 10 requirements')
plt.xlabel('Count')
plt.ylabel('Requirement')
plt.show()

plt.barh(top_titles.index, top_titles.values)
plt.title('Top 10 job titles')
plt.xlabel('Count')
plt.ylabel('Title')
plt.show()

plt.barh(top_locations.index, top_locations.values)
plt.title('Top 10 job locations')
plt.xlabel('Count')
plt.ylabel('Location')
plt.show()

# Use wordcloud to create a word cloud that shows the most common requirements and skills mentioned in the job descriptions
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['requirements']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Create a report in either Markdown or Jupyter Notebook format that summarizes the key findings of your analysis and includes the visualizations you created