import streamlit as st

from ml_app import run_ml_app

def main():
    menu = ['Home', 'Find Customer Cluster']
    choice = st.sidebar.radio("Menu", menu)

    if choice == 'Home':
            st.markdown(
            '''
            <h1 style='text-align: center;'> Discover Hidden Patterns Among Your Customers </h1>
            <br>
            <h4 style='text-align: left;'> Improve Business Strategy with Customer Clustering </h4>
            <p style='text-align: justify;'> With this tool, you can uncover deep insights from your customer data. By utilizing machine learning technology, this tool allows you  :
                <ul style='text-align: justify;'>
                    <li><strong>Identify Customer Segments:</strong> Separate customers into clusters based on their personal data, so you can target more specific and effective marketing strategies. </li>
                    <li><strong>Analyze Data Efficiently:</strong> A fast analytics process allows you to save time while ensuring accurate results, even if your data is large or complex. </li>
                    <li><strong>Improve Strategic Decisions:</strong> These tools not only provide you with data, but also actionable insights to support better decision-making. </li>
                </ul>
            </p>
            <br>
            <p style='text-align: justify;'><strong>Disclaimer:</strong> This tool is only to help you in analyzing customer data and may make analysis errors. Do further analysis according to your business case. </p>
            <br>
            <p style='text-align: center;'><strong>Discover the hidden potential in your customer data and take your business to the next level together.</strong></p>
            ''',
            unsafe_allow_html=True
        )
    elif choice == 'Find Customer Cluster':
        run_ml_app()


if __name__ == '__main__':
    main()
