# Importing necessary Libraries -------------------------------------------------------------------------
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
cv_vectorizer = CountVectorizer()
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st

# Unpacking stored data
job_data = pd.DataFrame(pickle.load(open('job_data.pkl','rb')))

# Defining a function which takes user input and returns most similar data --------------------------------
def similar(text):
    lower_converted = text.lower()
    list_of_words = nltk.word_tokenize(lower_converted)
    only_alpha_numeric_words = []
    for i in list_of_words:
        if i.isalnum():
            only_alpha_numeric_words.append(i)
    cleaned_text = []
    for i in only_alpha_numeric_words:
        if i not in stopwords.words('english') and i not in string.punctuation:
            cleaned_text.append(i)
    final_text = []
    for i in cleaned_text:
        final_text.append(ps.stem(i))
    preprocessed_text =  " ".join(final_text)
    x = cv_vectorizer.fit_transform(job_data['tags'])
    user_input_vector = cv_vectorizer.transform([preprocessed_text])
    cosine_similarities = cosine_similarity(user_input_vector, x).flatten()
    job_data['cv_similarity'] = cosine_similarities
    primary_goal = job_data.sort_values(by = 'cv_similarity',ascending = False)
    head = 0
    for i in range(len(primary_goal)):
        current_diff = primary_goal.iloc[0]['cv_similarity'] - primary_goal.iloc[i]['cv_similarity']
        if current_diff <= 0.3:
            head = i
        else:
            break
    all_job_openings = primary_goal.head(head)
    dict = {
            "Most Common Experience Level": all_job_openings['experience'].mode()[0],
            "Most Common Location": all_job_openings['job_locations'].mode()[0],
            "Most Common Company Class": all_job_openings['company_class'].mode()[0],
            "Total Number of Job Openings": len(all_job_openings)
                }

    primary_goal = pd.DataFrame(dict,index = [0]).T.reset_index()
    primary_goal = primary_goal.rename(columns={"index":"Key",0:"Insights"})

    return primary_goal, all_job_openings


# Building Frontend ------------------------------------------------------------------------------------

st.title("Job Finder System")
text = st.text_area("Search jobs by title, location, or skills")
no_of_results = st.slider("How many results you want to see?",0,100,5)
if st.button("Search"):
    primary_goal, all_job_openings = similar(text)
    st.subheader("Key-Insights for you search -")
    st.table(primary_goal)
    if no_of_results > len(all_job_openings):
        no_of_results = len(all_job_openings)
        st.text(str(no_of_results)+" results found only. Showing maximum results.")
    st.divider()
    #################################### Jobs Display ####################################
    for i in range(no_of_results):
        col1, col2 = st.columns(2)
        with col1:
            st.header(all_job_openings['company_name'].iloc[i])
            st.image(all_job_openings['Image_link'].iloc[i])
            st.text("A "+all_job_openings['company_class'].iloc[i]+" Company")
        with col2:
            st.text("Hiring for: "+all_job_openings['job_title'].iloc[i])
            st.text("Location/s: "+all_job_openings['job_locations'].iloc[i])
            st.text("Work Experience: "+all_job_openings['experience'].iloc[i])
            st.text("Skills: "+all_job_openings['skills'].iloc[i])
        if i==len(all_job_openings):
            break
    ##############################################################################
        st.divider()


    st.map(all_job_openings[['lat','lon']])

