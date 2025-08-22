import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Titanic Dataset Analysis",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and preprocess the Titanic dataset"""
    try:
        df = pd.read_csv('Titanic.csv')
        
        # Basic preprocessing
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
        
        # Create age groups
        df['AgeGroup'] = pd.cut(df['Age'], 
                               bins=[0, 12, 18, 35, 50, 100], 
                               labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'],
                               include_lowest=True)
        
        # Create fare groups
        df['FareGroup'] = pd.cut(df['Fare'], 
                                bins=[0, 10, 25, 50, 100, 1000], 
                                labels=['Low', 'Medium', 'High', 'Very High', 'Luxury'],
                                include_lowest=True)
        
        # Create family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Create family type
        df['FamilyType'] = df['FamilySize'].apply(lambda x: 'Alone' if x == 1 else 'Small' if x <= 3 else 'Large')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def display_overview(df):
    """Display dataset overview and basic statistics"""
    st.markdown('<h1 class="main-header">🚢 Titanic Dataset Analysis</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Passengers", len(df))
    
    with col2:
        st.metric("Survivors", df['Survived'].sum())
    
    with col3:
        survival_rate = (df['Survived'].sum() / len(df)) * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
    
    with col4:
        st.metric("Missing Age Values", df['Age'].isnull().sum())
    
    # Dataset info
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Dataset Info")
        buffer = st.empty()
        with buffer.container():
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Missing values
    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Percentage': (missing_data.values / len(df)) * 100
    })
    st.dataframe(missing_df, use_container_width=True)

def survival_analysis(df):
    """Analyze survival patterns"""
    st.markdown('<h2 class="section-header">🆘 Survival Analysis</h2>', unsafe_allow_html=True)
    
    # Overall survival statistics
    col1, col2 = st.columns(2)
    
    with col1:
        survival_counts = df['Survived'].value_counts()
        fig = px.pie(values=survival_counts.values, 
                    names=['Did Not Survive', 'Survived'],
                    title='Overall Survival Rate',
                    color_discrete_sequence=['#ff7f0e', '#2ca02c'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Survival by gender
        gender_survival = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
        fig = px.bar(gender_survival, 
                    title='Survival by Gender',
                    barmode='group',
                    color_discrete_sequence=['#ff7f0e', '#2ca02c'])
        fig.update_layout(xaxis_title='Gender', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    # Survival by passenger class
    st.subheader("Survival by Passenger Class")
    class_survival = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
    fig = px.bar(class_survival, 
                title='Survival by Passenger Class',
                barmode='group',
                color_discrete_sequence=['#ff7f0e', '#2ca02c'])
    fig.update_layout(xaxis_title='Passenger Class', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Survival by age group
    st.subheader("Survival by Age Group")
    age_survival = df.groupby(['AgeGroup', 'Survived']).size().unstack(fill_value=0)
    fig = px.bar(age_survival, 
                title='Survival by Age Group',
                barmode='group',
                color_discrete_sequence=['#ff7f0e', '#2ca02c'])
    fig.update_layout(xaxis_title='Age Group', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

def demographic_analysis(df):
    """Analyze demographic patterns"""
    st.markdown('<h2 class="section-header">👥 Demographic Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(df, x='Age', nbins=30, 
                          title='Age Distribution',
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(xaxis_title='Age', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_counts = df['Sex'].value_counts()
        fig = px.pie(values=gender_counts.values, 
                    names=gender_counts.index,
                    title='Gender Distribution',
                    color_discrete_sequence=['#ff7f0e', '#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Passenger class distribution
    st.subheader("Passenger Class Distribution")
    class_counts = df['Pclass'].value_counts().sort_index()
    fig = px.bar(x=class_counts.index, y=class_counts.values,
                title='Passenger Class Distribution',
                color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title='Passenger Class', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Family size analysis
    col1, col2 = st.columns(2)
    
    with col1:
        family_size_counts = df['FamilySize'].value_counts().sort_index()
        fig = px.bar(x=family_size_counts.index, y=family_size_counts.values,
                    title='Family Size Distribution',
                    color_discrete_sequence=['#2ca02c'])
        fig.update_layout(xaxis_title='Family Size', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        family_type_counts = df['FamilyType'].value_counts()
        fig = px.pie(values=family_type_counts.values, 
                    names=family_type_counts.index,
                    title='Family Type Distribution',
                    color_discrete_sequence=['#d62728', '#9467bd', '#8c564b'])
        st.plotly_chart(fig, use_container_width=True)

def fare_analysis(df):
    """Analyze fare patterns"""
    st.markdown('<h2 class="section-header">💰 Fare Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fare distribution
        fig = px.histogram(df, x='Fare', nbins=30, 
                          title='Fare Distribution',
                          color_discrete_sequence=['#ff7f0e'])
        fig.update_layout(xaxis_title='Fare', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fare by passenger class
        fig = px.box(df, x='Pclass', y='Fare', 
                    title='Fare Distribution by Passenger Class',
                    color_discrete_sequence=['#1f77b4'])
        fig.update_layout(xaxis_title='Passenger Class', yaxis_title='Fare')
        st.plotly_chart(fig, use_container_width=True)
    
    # Fare by embarkation port
    st.subheader("Fare by Embarkation Port")
    fig = px.box(df, x='Embarked', y='Fare', 
                title='Fare Distribution by Embarkation Port',
                color_discrete_sequence=['#2ca02c'])
    fig.update_layout(xaxis_title='Embarkation Port', yaxis_title='Fare')
    st.plotly_chart(fig, use_container_width=True)

def correlation_analysis(df):
    """Analyze correlations between variables"""
    st.markdown('<h2 class="section-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # Correlation heatmap
    fig = px.imshow(correlation_matrix,
                    title='Correlation Matrix Heatmap',
                    color_continuous_scale='RdBu',
                    aspect='auto')
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed correlation values
    st.subheader("Correlation Values")
    st.dataframe(correlation_matrix.round(3), use_container_width=True)

def machine_learning_analysis(df):
    """Perform machine learning analysis for survival prediction"""
    st.markdown('<h2 class="section-header">🤖 Machine Learning Analysis</h2>', unsafe_allow_html=True)
    
    # Prepare data for ML
    ml_df = df.copy()
    
    # Handle missing values
    ml_df['Age'].fillna(ml_df['Age'].median(), inplace=True)
    ml_df['Fare'].fillna(ml_df['Fare'].median(), inplace=True)
    ml_df['Embarked'].fillna(ml_df['Embarked'].mode()[0], inplace=True)
    
    # Encode categorical variables
    ml_df['Sex'] = ml_df['Sex'].map({'male': 0, 'female': 1})
    ml_df['Embarked'] = ml_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = ml_df[features]
    y = ml_df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature',
                    title='Feature Importance',
                    orientation='h',
                    color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm,
                        title='Confusion Matrix',
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Not Survived', 'Survived'],
                        y=['Not Survived', 'Survived'],
                        color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

def interactive_analysis(df):
    """Interactive analysis section"""
    st.markdown('<h2 class="section-header">🎯 Interactive Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by passenger class
        selected_class = st.selectbox("Select Passenger Class:", [1, 2, 3])
        class_filtered = df[df['Pclass'] == selected_class]
        
        st.write(f"**Passengers in Class {selected_class}:** {len(class_filtered)}")
        st.write(f"**Survival Rate:** {(class_filtered['Survived'].sum() / len(class_filtered) * 100):.1f}%")
        
        # Age distribution for selected class
        fig = px.histogram(class_filtered, x='Age', nbins=20,
                          title=f'Age Distribution - Class {selected_class}',
                          color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Filter by age group
        selected_age_group = st.selectbox("Select Age Group:", df['AgeGroup'].dropna().unique())
        age_filtered = df[df['AgeGroup'] == selected_age_group]
        
        st.write(f"**Passengers in {selected_age_group} group:** {len(age_filtered)}")
        st.write(f"**Survival Rate:** {(age_filtered['Survived'].sum() / len(age_filtered) * 100):.1f}%")
        
        # Gender distribution for selected age group
        gender_counts = age_filtered['Sex'].value_counts()
        fig = px.pie(values=gender_counts.values, 
                    names=gender_counts.index,
                    title=f'Gender Distribution - {selected_age_group}',
                    color_discrete_sequence=['#ff7f0e', '#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the Streamlit app"""
    st.sidebar.title("🚢 Titanic Analysis")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if Titanic.csv exists.")
        return
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose Analysis Section:",
        ["Overview", "Survival Analysis", "Demographics", "Fare Analysis", 
         "Correlations", "Machine Learning", "Interactive Analysis"]
    )
    
    # Display selected page
    if page == "Overview":
        display_overview(df)
    elif page == "Survival Analysis":
        survival_analysis(df)
    elif page == "Demographics":
        demographic_analysis(df)
    elif page == "Fare Analysis":
        fare_analysis(df)
    elif page == "Correlations":
        correlation_analysis(df)
    elif page == "Machine Learning":
        machine_learning_analysis(df)
    elif page == "Interactive Analysis":
        interactive_analysis(df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🚢 Titanic Dataset Analysis | Built with Streamlit</p>
            <p>Data Source: Titanic passenger dataset</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
