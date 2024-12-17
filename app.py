import streamlit as st
from dotenv import load_dotenv
import base64
import os
import io
import fitz  # PyMuPDF
from PIL import Image
import pdf2image
import google.generativeai as genai
import matplotlib.pyplot as plt
import re
import requests
from bs4 import BeautifulSoup

# Set Streamlit page config (must be first Streamlit command)
st.set_page_config(page_title="ATS Resume System", page_icon=":bar_chart:", layout="wide")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    try:
        document = fitz.open(stream=uploaded_file.read(), filetype='pdf')
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def get_gemini_response(input_text, resume_text, linkedin_content, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        combined_input = f"Input Text: {input_text}\n\nResume Content: {resume_text}\n\nLinkedIn Content: {linkedin_content}\n\nPrompt: {prompt}"
        response = model.generate_content(combined_input)
        response_text = response.text

        # Modify the response to exclude the unwanted portion
        unwanted_portion = "The Response is"
        if unwanted_portion in response_text:
            response_text = response_text.split(unwanted_portion)[-1].strip()
        
        return response_text
    except Exception as e:
        st.error(f"Error generating response from AI model: {e}")
        return ""

def input_pdf_setup(uploaded_file):
    try:
        image = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = image[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    except Exception as e:
        st.error(f"Error converting PDF to image: {e}")
        return []

def visualize_results(results):
    categories = ['Skills Match', 'Experience Match', 'Education Match', 'Keywords Match', 'Projects and Achievements Match']
    scores = [results['skills'], results['experience'], results['education'], results['keywords'], results['projects']]
    
    fig, ax = plt.subplots()
    ax.bar(categories, scores, color='skyblue')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Scores (%)')
    ax.set_title('Resume Match Analysis')
    
    st.pyplot(fig)

def extract_percentage(response):
    match = re.search(r'(\d+)%', response)
    if match:
        return match.group(1)
    return "N/A"

def scrape_linkedin_profile(linkedin_url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(linkedin_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        linkedin_content = soup.get_text(separator="\n")
        return linkedin_content
    except Exception as e:
        st.error(f"Error scraping LinkedIn profile: {e}")
        return ""

def process_resume(uploaded_file, input_text, input_prompt, linkedin_url):
    resume_text = extract_text_from_pdf(uploaded_file)
    linkedin_content = scrape_linkedin_profile(linkedin_url)
    response = get_gemini_response(input_text, resume_text, linkedin_content, input_prompt)
    return response

# Title with logo on the side
col1, col2 = st.columns([8, 1])
with col1:
    st.title("ATS Resume System")
with col2:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQn34fT7shq_xoPwkbNaLAt7JQXg6ULi49RNg&s", width=150)

# User input for job description
input_text = st.text_area("Job Description:", key="input")

# User input for LinkedIn profile URL
linkedin_url = st.text_input("LinkedIn Profile URL:", key="linkedin")

# File uploader for resume
uploaded_file = st.file_uploader("Upload your Resume", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF Uploaded Successfully")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        Submit_1 = st.button("Tell me about the resume", key="button1")
    with col2:
        Submit_2 = st.button("Percentage Match", key="button2")
    with col3:
        Submit_3 = st.button("Skill Gaps Analysis", key="button3")
    with col1:
        Submit_4 = st.button("Project Evaluation", key="button4")
    with col2:
        Submit_5 = st.button("Cultural Fit Assessment", key="button5")
    with col3:
        Submit_6 = st.button("Personalized Learning Path", key="button6")

    input_prompt_1 = """
        You are an experienced HR profes
        sional with technical expertise in Data Science, Full Stack Development, Web Development, or Machine Learning.

        Your task is to review the provided resume against the specified job description for one of these profiles. Please perform a thorough evaluation and provide a professional assessment based on the following criteria:

        1. **Alignment with Job Requirements**: Evaluate how well the candidate's experience, skills, and qualifications align with the job description.
        2. **Technical Skills**: Assess the candidate's technical skills relevant to the job profile, including programming languages, tools, and technologies.
        3. **Professional Experience**: Review the candidate's previous job roles, responsibilities, and achievements to determine their suitability for the position.
        4. **Educational Background**: Examine the candidate's educational qualifications, certifications, and any relevant coursework.
        5. **Soft Skills**: Evaluate the candidate's communication, teamwork, problem-solving abilities, and other soft skills mentioned in the resume.
        6. **Strengths and Weaknesses**: Highlight the candidate's key strengths and any areas where they may fall short in relation to the job requirements.
        7. **Overall Recommendation**: Provide an overall recommendation on the candidate's suitability for the role.

        Please structure your evaluation clearly and concisely, providing specific examples from the resume to support your assessment.
    """
    
    input_prompt_2 = """
        You are an AI specializing in resume evaluation and job matching. Your task is to analyze the uploaded resume and the provided job description for a specific role in Data Science, Full Stack Development, Web Development, or Machine Learning.

        Please calculate the percentage match between the resume and the job description based on the following criteria:
        1. Skills: Match the technical and soft skills listed in the job description with those mentioned in the resume.
        2. Experience: Compare the candidate's work experience, including the relevance and duration, with the job requirements.
        3. Education: Evaluate the educational background and certifications mentioned in the resume against the job qualifications.
        4. Keywords: Identify and match specific keywords and phrases from the job description with those present in the resume.
        5. Projects and Achievements: Assess the relevance and impact of projects and achievements listed in the resume to the job role.

        Provide a detailed evaluation and the final percentage match, along with a summary of strengths and areas for improvement.

        Example format:
        - Skills Match: 90%
        - Experience Match: 85%
        - Education Match: 80%
        - Keywords Match: 75%
        - Projects and Achievements Match: 70%
        - Overall Match: 80%
    """

    input_prompt_3 = """
        You are an AI specializing in identifying skill gaps. Your task is to analyze the uploaded resume and the provided job description for a specific role in Data Science, Full Stack Development, Web Development, or Machine Learning.

        Please identify any gaps in the candidate's skills relative to the job description. Highlight specific skills that are required by the job but are missing or insufficiently covered in the resume.

        Provide actionable suggestions for the candidate to fill these gaps through training, certification, or practical experience.
    """

    input_prompt_4 = """
        You are an AI specializing in evaluating project quality and relevance. Your task is to analyze the projects listed in the uploaded resume and assess their relevance to the provided job description for a specific role in Data Science, Full Stack Development, Web Development, or Machine Learning.

        Please evaluate the relevance and quality of the projects based on the following criteria:
        1. Project Relevance: How closely the projects align with the job requirements.
        2. Project Complexity: The complexity and scope of the projects.
        3. Project Impact: The impact and outcomes of the projects.

        Provide a detailed evaluation of each project, highlighting strengths and areas for improvement.
    """

    input_prompt_5 = """
        You are an AI specializing in cultural fit assessment. Your task is to evaluate the candidate's cultural fit for the provided job description and company values for a specific role in Data Science, Full Stack Development, Web Development, or Machine Learning.

        Please analyze the candidate's resume and job description to assess their fit with the company's culture and values. Consider the following criteria:
        1. Alignment with Company Values: How well the candidate's values align with the company's values.
        2. Work Environment Compatibility: The candidate's fit with the company's work environment and team dynamics.
        3. Potential Contributions: How the candidate can contribute to and thrive in the company's culture.

        Provide a detailed assessment, highlighting strengths and areas for improvement in terms of cultural fit.
    """



    input_prompt_6 = """
        You are an AI specializing in creating personalized learning paths.
        Your task is to analyze the uploaded resume and the provided job description for a specific role in Data Science, Full Stack Development, Web Development, or Machine Learning.
        Based on the analysis, identify the candidate's skill gaps and provide a personalized learning path with specific recommendations for courses, certifications, or skills development.
        Please consider the following criteria:
        1. **Skill Gaps Identification**: Highlight the skills required for the job that are missing or insufficient in the candidate's resume. 
        2. **Personalized Recommendations**: Suggest specific courses, certifications, or practical experiences that can help the candidate fill these skill gaps. 
        3. **Actionable Steps**: Provide actionable steps the candidate can take to improve their qualifications and increase their chances of getting the job.
        
        Ensure the recommendations are relevant and practical, offering specific examples and resources.
    """
    if Submit_1:
        response = process_resume(uploaded_file, input_text, input_prompt_1, linkedin_url)
        st.subheader("The Response is")
        st.write(response)
    
    if Submit_2:
        response = process_resume(uploaded_file, input_text, input_prompt_2, linkedin_url)
        st.subheader("The Response is")
        st.write(response)
        # Visualize results if available in the response
        # visualize_results(results)
    
    if Submit_3:
        response = process_resume(uploaded_file, input_text, input_prompt_3, linkedin_url)
        st.subheader("The Response is")
        st.write(response)

    if Submit_4:
        response = process_resume(uploaded_file, input_text, input_prompt_4, linkedin_url)
        st.subheader("The Response is")
        st.write(response)
    
    if Submit_5:
        response = process_resume(uploaded_file, input_text, input_prompt_5, linkedin_url)
        st.subheader("The Response is")
        st.write(response)
    
    if Submit_6:
        response = process_resume(uploaded_file, input_text, input_prompt_6, linkedin_url)
        st.subheader("Personalized Learning Path")
        st.write(response)
