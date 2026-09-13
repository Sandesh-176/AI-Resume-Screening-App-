import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity # Measure two texts similarity 

nltk.download('punkt')
nltk.download('stopwords')

# loding models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

def CleanResume(resume_text):
    cleanTxt = re.sub('http\S+\s'," ",resume_text) #find URLs or HTTP links in text so they can be removed during text cleaning.
    cleanTxt = re.sub('@\S+'," ",cleanTxt) #Removing mentions/usernames
    cleanTxt = re.sub('#\S+'," ",cleanTxt) #Removing hashtags
    cleanTxt = re.sub('RT|CC'," ",cleanTxt) #Removing the words RT or CC
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")," ",cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]'," ",cleanTxt) #Removing non-ASCII characters. ' é '
    cleanTxt = re.sub('\s+'," ",cleanTxt) #Removing/replacing multiple spaces, tabs, and new lines.
    return cleanTxt


# SKILL DATABASE
skills_list = [
    # PROGRAMMING LANGUAGES
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "php",
    "ruby",
    "go",
    "kotlin",
    "swift",
    "r",
    "scala",

    # WEB DEVELOPMENT
    "html",
    "css",
    "bootstrap",
    "tailwind css",
    "javascript",
    "jquery",
    "react",
    "react.js",
    "angular",
    "vue.js",
    "node.js",
    "express.js",
    "next.js",
    "django",
    "flask",
    "fastapi",

    # JAVA TECHNOLOGIES
    "core java",
    "java ee",
    "servlet",
    "jsp",
    "jstl",
    "spring",
    "spring mvc",
    "spring boot",
    "spring security",
    "hibernate",
    "jpa",
    "maven",
    "gradle",
    "jdbc",

    # DATABASES
    "sql",
    "mysql",
    "postgresql",
    "oracle",
    "sqlite",
    "mongodb",
    "redis",
    "cassandra",
    "firebase",
    "pl/sql",

    # DATA SCIENCE
    "data science",
    "data analysis",
    "data analytics",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "nlp",
    "computer vision",

    # PYTHON DATA LIBRARIES
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "scipy",
    "tensorflow",
    "pytorch",
    "keras",
    "opencv",

    # MACHINE LEARNING
    "linear regression",
    "logistic regression",
    "decision tree",
    "random forest",
    "support vector machine",
    "svm",
    "knn",
    "k-nearest neighbors",
    "naive bayes",
    "k-means",
    "clustering",
    "xgboost",
    "lightgbm",
    "feature engineering",
    "model evaluation",

    # GENERATIVE AI / LLM
    "generative ai",
    "large language model",
    "llm",
    "chatgpt",
    "openai",
    "transformers",
    "hugging face",
    "langchain",
    "prompt engineering",
    "rag",
    "retrieval augmented generation",

    # CLOUD
    "aws",
    "amazon web services",
    "azure",
    "google cloud",
    "gcp",
    "ec2",
    "s3",
    "lambda",
    "cloud computing",

    # DEVOPS
    "git",
    "github",
    "gitlab",
    "docker",
    "kubernetes",
    "jenkins",
    "ci/cd",
    "continuous integration",
    "continuous deployment",
    "terraform",
    "ansible",
    "linux",
    "shell scripting",
    "bash",

    # DATA VISUALIZATION
    "power bi",
    "tableau",
    "excel",
    "advanced excel",
    "power query",
    "dax",
    "looker",

    # BIG DATA
    "hadoop",
    "spark",
    "apache spark",
    "hive",
    "pig",
    "kafka",
    "databricks",
    "etl",
    "data warehouse",

    # API / BACKEND
    "rest api",
    "restful api",
    "web services",
    "soap",
    "json",
    "xml",
    "microservices",
    "api development",

    # TESTING
    "software testing",
    "manual testing",
    "automation testing",
    "selenium",
    "testng",
    "junit",
    "pytest",
    "cypress",
    "postman",
    "api testing",
    "unit testing",
    "integration testing",

    # CYBERSECURITY
    "cybersecurity",
    "network security",
    "ethical hacking",
    "penetration testing",
    "information security",
    "cryptography",
    "firewall",
    "siem",

    # PROJECT / DEVELOPMENT TOOLS
    "agile",
    "scrum",
    "jira",
    "confluence",
    "github actions",
    "visual studio code",
    "jupyter notebook",
    "anaconda",

    # SOFT SKILLS
    "communication",
    "leadership",
    "teamwork",
    "problem solving",
    "time management",
    "critical thinking",
    "project management"
]

# EXTRACT SKILLS FROM TEXT
def extract_skills(text):
    text = text.lower()
    found_skills = []

    # Check longer/multi-word skills first
    sorted_skills = sorted(skills_list,key=len,reverse=True)
    for skill in sorted_skills:
        escaped_skill = re.escape(skill.lower())
        pattern = r'(?<!\w)' + escaped_skill + r'(?!\w)'
        if re.search(pattern, text):
            found_skills.append(skill)
    return found_skills

resume_sections = [
    "education",
    "experience",
    "work experience",
    "skills",
    "projects",
    "certifications",
    "summary",
    "objective"
]

# Map category ID to category name
category_mapping = {
    15 : "Java Developer",
    23 : "Testing",
    8 : "DevOps Engineer",
    20 : "Python Developer",
    24 : "Web Designing",
    12 : "HR",
    13 : "Hadoop",
    3 : "Blockchain",
    10 : "ETL Developer",
    18 : "Operations Manager",
    6 : "Data science",
    22 : "sales",
    16 : "Machine Learning",
    1 : "Arts",
    7 : "Database",
    11 : "Electrical Engineering",
    14 : "Health and Fitness",
    19 : "PMO",
    4 : "Busines Analysy",
    9 : "DotNet Developer",
    2 : "Automation Testing",
    17 : "Network Security Engineer",
    21 : "SAP Developer",
    5 : "Civil Engineer",
    0 : "Advocate"
}

    
# ============================================================
# WEB APP
# ============================================================

def main():

    st.title("AI Resume Screening System")

    st.write(
        "Upload a resume and enter a job description "
        "to analyze the candidate."
    )

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "txt"]
    )

    job_description = st.text_area(
        "Enter Job Description",
        height=250,
        placeholder=(
            "Example: We are looking for a Java Developer "
            "with experience in Java, Spring Boot, Hibernate, "
            "SQL, REST API and Git."
        )
    )

    # ========================================================
    # ANALYZE RESUME BUTTON
    # ========================================================

    if st.button("Analyze Resume"):

        # ====================================================
        # CHECK RESUME
        # ====================================================

        if uploaded_file is None:

            st.warning("Please upload your resume.")

        # ====================================================
        # CHECK JOB DESCRIPTION
        # ====================================================

        elif job_description.strip() == "":

            st.warning("Please enter the job description.")

        # ====================================================
        # PROCESS RESUME
        # ====================================================

        else:

            try:

                # =================================================
                # PDF FILE
                # =================================================

                if uploaded_file.type == "application/pdf":

                    pdf_reader = PdfReader(uploaded_file)

                    resume_text = ""

                    for page in pdf_reader.pages:

                        text = page.extract_text()

                        if text:

                            resume_text += text + "\n"

                # =================================================
                # TXT FILE
                # =================================================

                else:

                    resume_bytes = uploaded_file.read()

                    try:

                        resume_text = resume_bytes.decode("utf-8")

                    except UnicodeDecodeError:

                        resume_text = resume_bytes.decode("latin-1")

                # =================================================
                # CHECK EXTRACTED TEXT
                # =================================================

                if not resume_text.strip():

                    st.error(
                        "Could not extract text from this resume."
                    )

                    st.info(
                        "Please upload a text-based PDF or TXT file."
                    )

                else:

                    # =================================================
                    # CLEAN RESUME
                    # =================================================

                    cleaned_resume = CleanResume(resume_text)


                    # =================================================
                    # EXTRACT SKILLS FROM RESUME
                    # =================================================

                    resume_skills = extract_skills(resume_text)


                    # =================================================
                    # EXTRACT SKILLS FROM JOB DESCRIPTION
                    # =================================================

                    jd_skills = extract_skills(job_description)


                    # =================================================
                    # FIND MATCHED SKILLS
                    # =================================================

                    matched_skills = list(
                        set(resume_skills) & set(jd_skills)
                    )


                    # =================================================
                    # FIND MISSING SKILLS
                    # =================================================

                    missing_skills = list(
                        set(jd_skills) - set(resume_skills)
                    )


                    # =================================================
                    # CALCULATE SKILL MATCH PERCENTAGE
                    # =================================================

                    if len(jd_skills) > 0:

                        skill_match_percentage = (
                            len(matched_skills)
                            / len(jd_skills)
                        ) * 100

                    else:

                        skill_match_percentage = 0


                    # =================================================
                    # RESUME - JOB DESCRIPTION SIMILARITY
                    # =================================================

                    resume_vector = tfidf.transform(
                        [cleaned_resume]
                    )

                    jd_cleaned = CleanResume(
                        job_description
                    )

                    jd_vector = tfidf.transform(
                        [jd_cleaned]
                    )

                    similarity = cosine_similarity(
                        resume_vector,
                        jd_vector
                    )[0][0]

                    similarity_percentage = similarity * 100


                    # =================================================
                    # RESUME COMPLETENESS
                    # =================================================

                    resume_lower = resume_text.lower()

                    found_sections = []

                    for section in resume_sections:

                        if section in resume_lower:

                            found_sections.append(section)


                    if len(resume_sections) > 0:

                        completeness_percentage = (
                            len(found_sections)
                            / len(resume_sections)
                        ) * 100

                    else:

                        completeness_percentage = 0


                    # =================================================
                    # CALCULATE ATS SCORE
                    # =================================================

                    ats_score = (
                        (skill_match_percentage * 0.50)
                        +
                        (similarity_percentage * 0.30)
                        +
                        (completeness_percentage * 0.20)
                    )


                    # =================================================
                    # SUCCESS MESSAGE
                    # =================================================

                    st.success(
                        "Resume and Job Description "
                        "received successfully."
                    )


                    # =================================================
                    # PREDICT CATEGORY
                    # =================================================

                    input_features = tfidf.transform(
                        [cleaned_resume]
                    )

                    prediction_id = clf.predict(
                        input_features
                    )[0]

                    category_name = category_mapping.get(
                        prediction_id,
                        "Unknown"
                    )

                    st.success(
                        f"Predicted Category: {category_name}"
                    )


                    # =================================================
                    # ATS SCORE
                    # =================================================

                    st.subheader("ATS Score")

                    st.metric(
                        label="Overall ATS Score",
                        value=f"{ats_score:.2f}%"
                    )

                    st.progress(
                        int(ats_score)
                    )


                    # =================================================
                    # ATS SCORE COMPONENTS
                    # =================================================

                    col1, col2, col3 = st.columns(3)


                    with col1:

                        st.metric(
                            "Skill Match",
                            f"{skill_match_percentage:.2f}%"
                        )


                    with col2:

                        st.metric(
                            "JD Similarity",
                            f"{similarity_percentage:.2f}%"
                        )


                    with col3:

                        st.metric(
                            "Resume Completeness",
                            f"{completeness_percentage:.2f}%"
                        )


                    # =================================================
                    # SKILL ANALYSIS
                    # =================================================

                    st.subheader("Skill Analysis")

                    col1, col2 = st.columns(2)


                    # =================================================
                    # MATCHED SKILLS
                    # =================================================

                    with col1:

                        st.write("### ✅ Matched Skills")

                        if matched_skills:

                            for skill in sorted(
                                matched_skills
                            ):

                                st.success(
                                    skill.title()
                                )

                        else:

                            st.warning(
                                "No matching skills found."
                            )


                    # =================================================
                    # MISSING SKILLS
                    # =================================================

                    with col2:

                        st.write("### ❌ Missing Skills")

                        if missing_skills:

                            for skill in sorted(
                                missing_skills
                            ):

                                st.error(
                                    skill.title()
                                )

                        else:

                            st.success(
                                "No required skills are missing!"
                            )


                    # =================================================
                    # SKILL MATCH SCORE
                    # =================================================

                    st.subheader(
                        "Skill Match Score"
                    )

                    st.progress(
                        int(skill_match_percentage)
                    )

                    st.write(
                        f"**{skill_match_percentage:.2f}%** "
                        "of the required skills were found "
                        "in the resume."
                    )


                    # =================================================
                    # RESUME-JD SIMILARITY
                    # =================================================

                    st.write(
                        f"Resume-JD Similarity: "
                        f"{similarity_percentage:.2f}%"
                    )


                    # =================================================
                    # EXTRACTED RESUME TEXT
                    # =================================================

                    with st.expander(
                        "View extracted resume text"
                    ):

                        st.write(
                            resume_text[:3000]
                        )


                    # =================================================
                    # PREDICTION DETAILS
                    # =================================================

                    with st.expander(
                        "View prediction details"
                    ):

                        st.write(
                            "Prediction ID:",
                            prediction_id
                        )

                        st.write(
                            "Number of characters:",
                            len(cleaned_resume)
                        )

                        st.write(
                            "Number of TF-IDF features:",
                            input_features.nnz
                        )


            except Exception as e:

                st.error(
                    f"Error processing resume: {e}"
                )

# PYTHON MAIN
if __name__ == "__main__":

    main()