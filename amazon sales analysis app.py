import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import plotly.express as px
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Amazon Sales Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    data = pd.read_csv(r'c:\Users\hp\Downloads\amazon\amazon.csv')
   
    data.columns = ['product_id', 'product_name', 'category', 'discounted_price', 'actual_price', 'discount_percentage', 
                    'rating', 'rating_count', 'about_product', 'user_id', 'user_name', 'review_id', 'review_title', 'review_content',
                    'img_link', 'product_link']  # Fixed missing comma
    data.loc[data["rating_count"].isnull(), 'rating_count'] = data["rating_count"].mode()[0]
    data["discounted_price"] = data["discounted_price"].apply(lambda x: x.replace("₹", "").replace(",", ""))
    data["actual_price"] = data["actual_price"].apply(lambda x: x.replace("₹", "").replace(",", ""))
    data["rating_count"] = data['rating_count'].apply(lambda x: x.replace(",", ""))
    data["discount_percentage"] = data["discount_percentage"].apply(lambda x: x.replace("%", ""))

    data["discounted_price"] = pd.to_numeric(data["discounted_price"], errors='coerce')
    data["actual_price"] = pd.to_numeric(data["actual_price"], errors='coerce')
    data["rating_count"] = pd.to_numeric(data["rating_count"], errors='coerce')
    data["discount_percentage"] = pd.to_numeric(data["discount_percentage"], errors='coerce')
    data["rating"] = pd.to_numeric(data["rating"], errors='coerce')
    data.loc[data['rating'].isnull(), 'rating'] = data['rating'].mode()[0]

    data['category'] = data['category'].str.split("|").str.get(0)
    data['product_type'] = data['category'].str.split("|").str.get(-1)
    data['total_sales'] = data['discounted_price'] * data['rating_count']
    return data

data = load_data()

# Sidebar navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",  # Required
        options=["Home", "Data Overview", "Exploratory Data Analysis", "Statistical Analysis", "Model", "Conclusion"],  # Menu options
        icons=["house", "clipboard-data", "bar-chart", "robot", "check-circle"],  # Optional icons (FontAwesome)
        menu_icon="cast",  # Icon for the menu title
        default_index=0,  # Index of the initially selected menu item
        styles={
            "container": {"padding": "5px", "background-color": "#f4f4f4"},  # Background for the sidebar
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "--hover-color": "#e0e0e0"},  # Non-selected items
            "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},  # Selected item
        },
    )

# Display content based on selected menu
if selected == "Home":
    st.title("🏠 Home")
    st.markdown("""
        Welcome to the **Amazon Sales Analysis Dashboard**. Use the sidebar navigation to explore various insights and machine learning models!
    """)
    st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" alt="Amazon Logo" style="width: 50%; max-width: 300px;">
        <p style="font-size: 16px; color: #555;">Amazon Logo</p>
    </div>
    """,
    unsafe_allow_html=True
)
    st.subheader("Introuction")  
    st.markdown("""
        Amazon, founded by Jeff Bezos in 1994, is one of the world's largest multinational technology companies. 
        Initially starting as an online bookstore, Amazon has grown into a global e-commerce giant, offering 
        a vast range of products, from electronics and clothing to groceries and digital services. 
        It is renowned for its customer-centric approach, innovative technologies, and extensive delivery network.
    """)

elif selected == "Data Overview":
    st.header("📋 Data Overview")
    st.subheader("Attributes in the Dataset")
    attribute_names = data.columns.tolist()
    attribute_df = pd.DataFrame(attribute_names, columns=["Attribute Names"])
    st.table(attribute_df)
    
    st.write(f"Total number of rows in the dataset: {len(data)}")
    
    st.subheader("Summary Statistics for Numerical Columns")
    numerical_columns = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'total_sales']
    summary_stats = data[numerical_columns].describe().T
    summary_stats.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    st.dataframe(summary_stats)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head(15))
    
    st.subheader("Dataset Information")
    info_df = pd.DataFrame({
        'Non-Null Count': data.notnull().sum(),
        'Data Type': data.dtypes
    })
    st.table(info_df)

elif selected == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    # Scatter Plot: Discounted Price vs Total Sales
    st.subheader("Summary Statistics")
    summary_stats = data.describe()
    st.write(summary_stats)

    # Display Unique Counts for Each Column
    st.subheader("Unique Counts for Each Column")
    unique_counts = data.nunique().reset_index()
    unique_counts.columns = ['Column Name', 'Unique Values Count']        
    st.table(unique_counts)

    st.subheader("Missing Value Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Values Heatmap", fontsize=16)
    st.pyplot(fig)
    
    # Calculate Missing Value Percentages
    st.subheader("Missing Value Percentage")
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    missing_percentage_df = missing_percentage[missing_percentage > 0].reset_index()
    missing_percentage_df.columns = ['Column Name', 'Missing Percentage']
    
    if not missing_percentage_df.empty:
        st.table(missing_percentage_df)
    else:
        st.write("No missing values in the dataset!")


    st.subheader("Data Types in the Dataset")

    st.table(data.dtypes.astype(str))


    st.subheader("Feature Distribution")
    st.write("Select a column from the dropdown below to visualize its histogram.")

    numerical_columns = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'total_sales']

    selected_column = st.selectbox("Select a column for Feature Distribution", numerical_columns)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data[selected_column], bins=20, kde=True, color='purple', ax=ax)
    ax.set_title(f"Distribution of {selected_column}", fontsize=16)
    ax.set_xlabel(selected_column)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Box Plots for Outlier Detection")
    st.write("Select a column from the dropdown below to visualize its box plot.")

# Define the numerical columns for box plots
    numerical_columns = ['discounted_price', 'actual_price', 'rating', 'rating_count']

# Create a dropdown (selectbox) for column selection
    selected_column = st.selectbox("Select a column for Box Plot", numerical_columns)

# Display the box plot for the selected column
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=data[selected_column], ax=ax)
    ax.set_title(f"Boxplot of {selected_column}", fontsize=16)
    st.pyplot(fig)


    st.subheader("Scatter Plot: Discounted Price vs Total Sales")
    scatter_fig = px.scatter(
        data,
        x='discounted_price',
        y='total_sales',
        color='product_type',
        title='Discounted Price vs Total Sales',
        labels={'discounted_price': 'Discounted Price', 'total_sales': 'Total Sales'},
        template='plotly_dark'
    )
    st.plotly_chart(scatter_fig)

    # Scatter Plot: Rating Count vs Discounted Percentage
    st.subheader("Scatter Plot: Rating Count vs Discounted Percentage")
    with st.container():
    # Adjust the figure size and layout for better display
        fig, ax = plt.subplots(figsize=(8, 5))  # Adjusted to a slightly larger size
        ax.scatter(data['rating_count'], data['discount_percentage'], color='lightcoral', alpha=0.7, edgecolor='black')
        ax.set_title("Rating Count vs Discounted Percentage", fontsize=14)
        ax.set_xlabel("Rating Count", fontsize=12)
        ax.set_ylabel("Discounted Percentage", fontsize=12)
        ax.grid(alpha=0.3)

    # Display the plot
    st.pyplot(fig)

    # Bar Plot: Total Sales by Top 10 Categories
    st.subheader("Bar Plot: Total Sales by Categories")
    category_sales = data.groupby('product_type')['total_sales'].sum().reset_index()
    category_sales_top = category_sales.nlargest(10, 'total_sales')

    bar_fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        category_sales_top['product_type'],
        category_sales_top['total_sales'],
        color='skyblue',
        edgecolor='black'
    )

    # Add text labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.0f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_title("Total Sales by Categories", fontsize=14)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Total Sales", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    st.pyplot(bar_fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numeric_columns = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'total_sales']
    correlation_matrix = data[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Correlation Matrix", fontsize=14)
    st.pyplot(fig)


elif selected == "Statistical Analysis":
    st.header("📈 Statistical Analysis")
    
    # Top Product Metrics Section with Dropdown
    st.subheader("Top Product Types by Metrics")

    # Define available metrics for dropdown
    metric_options = {
        "Total Sales": "total_sales",
        "Average Ratings": "rating",
        "Number of Reviews": "review_id",
        "Average Discount Percentage": "discount_percentage"
    }
    selected_metric = st.selectbox("Select a Metric", list(metric_options.keys()))

    # Plot based on selected metric
    if selected_metric == "Total Sales":
        # Top 10 Products by Total Sales
        top_sales = data.groupby('product_type')['total_sales'].sum()
        top_sales_sorted = top_sales.sort_values(ascending=False)
        top_10_products = top_sales_sorted.head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_products.index, y=top_10_products.values, palette='viridis')
        plt.title('Top 10 Products by Sales', fontsize=16)
        plt.xlabel('Product Name', fontsize=12)
        plt.ylabel('Total Sales', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    elif selected_metric == "Average Ratings":
        # Top 10 Products by Average Rating
        top_rated = data.groupby('product_type')['rating'].mean()
        top_rated_sorted = top_rated.sort_values(ascending=False)
        top_10_rated = top_rated_sorted.head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_rated.index, y=top_10_rated.values, palette='viridis')
        plt.title('Top 10 Products by Ratings', fontsize=16)
        plt.xlabel('Product Name', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    elif selected_metric == "Number of Reviews":
        # Top 10 Products by Number of Reviews
        top_reviewed = data.groupby('product_type')['review_id'].count().reset_index()
        top_reviewed_sorted = top_reviewed.sort_values(by='review_id', ascending=False)
        top_10_reviewed = top_reviewed_sorted.head(10)

        fig = px.bar(top_10_reviewed, x='product_type', y='review_id',
                     title="Top 10 Products with the Most Reviews",
                     labels={'product_name': 'Product Name', 'review_id': 'Number of Reviews'},
                     template="plotly_dark")
        st.plotly_chart(fig)

    elif selected_metric == "Average Discount Percentage":
        # Top 10 Products by Average Discount Percentage
        top_discounts = data.groupby('product_type')['discount_percentage'].mean().sort_values(ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(range(len(top_discounts)))
        plt.bar(top_discounts.index, top_discounts.values, color=colors)
        plt.title('Top Product Types with Highest Discount Percentage', fontsize=14, fontweight='bold')
        plt.xlabel('Product Type', fontsize=12)
        plt.ylabel('Average Discount Percentage (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)

        # Add values on top of the bars
        for index, value in enumerate(top_discounts.values):
            plt.text(index, value + 0.5, f'{value:.2f}%', ha='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(plt)
    st.subheader("Hypothesis Testing: Impact of Discounts on Total Sales")

    low_discount = data[data['discount_percentage'] < 20]['total_sales']
    medium_discount = data[(data['discount_percentage'] >= 20) & (data['discount_percentage'] <= 50)]['total_sales']
    high_discount = data[data['discount_percentage'] > 50]['total_sales']

# Perform ANOVA test
    f_stat, p_val = f_oneway(low_discount, medium_discount, high_discount)

# Display the results
    st.write(f"F-statistic: {f_stat:.2f}")
    st.write(f"P-value: {p_val:.4f}")

    if p_val < 0.05:
        st.write("Significant impact: Discounts have a noticeable effect on total sales.")
    else:
        st.write("No significant impact: Discounts may not strongly influence total sales.")

# Visualize the impact using a mean line chart
    st.subheader("Visualizing the Impact of Discounts on Total Sales")

# Combine data into a single DataFrame for visualization
    data['discount_category'] = pd.cut(
    data['discount_percentage'], 
    bins=[0, 20, 50, 100], 
    labels=['Low (<20%)', 'Medium (20%-50%)', 'High (>50%)']
)

# Create the point plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.pointplot(
    x='discount_category', 
    y='total_sales', 
    data=data, 
    ci=95,  # 95% confidence interval
    markers='o',
    linestyles='-', 
    capsize=0.1, 
    color='royalblue',
    ax=ax
)

# Customize the chart
    ax.set_title('Mean Total Sales by Discount Category', fontsize=12)
    ax.set_xlabel('Discount Category', fontsize=9)
    ax.set_ylabel('Mean Total Sales', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Number of Products per Category")

# Create the figure
    plt.figure(figsize=(7, 6))
    ax = sns.countplot(data=data, x='category', palette="tab10")

# Add plot details
    plt.title('Number of Products per Category', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Product Count', fontsize=12)
    plt.xticks(rotation=90)

# Annotate bars with counts
    for p in ax.patches:
        ax.annotate(
        format(p.get_height(), '.0f'), 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='center', 
        xytext=(0, 9), 
        textcoords='offset points'
    )

# Ensure layout is clean
    plt.tight_layout()

# Display the plot in Streamlit
    st.pyplot(plt)



elif selected == "Model":
    st.header("🤖 Machine Learning Model: Predicting Total Sales")

    # Display model information
    st.subheader("Model: Random Forest Regressor")
    st.markdown("This section demonstrates the prediction of total sales using product features like product type, rating, rating count, discount percentage, and actual price.")

    # Prepare the dataset
    df_ml = data[['product_type', 'rating', 'rating_count', 'discount_percentage', 'actual_price', 'total_sales']].dropna()

    # Encode the categorical 'product_type'
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df_ml['product_type'] = label_encoder.fit_transform(df_ml['product_type'])

    # Define the features (X) and target variable (y)
    X = df_ml[['product_type', 'rating', 'rating_count', 'discount_percentage', 'actual_price']]  # Features
    y = df_ml['total_sales']  # Target variable

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestRegressor model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation Metrics")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Plot Actual vs Predicted values
    st.subheader("Actual vs Predicted Total Sales")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    ax.set_xlabel('Actual Total Sales')
    ax.set_ylabel('Predicted Total Sales')
    ax.set_title('Actual vs Predicted Total Sales')
    st.pyplot(fig)

    # Feature Importance Analysis
    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names, feature_importances, color='skyblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance in Predicting Total Sales')
    st.pyplot(fig)


elif selected == "Conclusion":
    st.header("✅ Conclusion")
    st.write("Summary of key insights and final takeaways.")
    st.markdown("""
Through this analysis, we uncovered valuable insights into Amazon's sales dynamics. Key findings include:

1. **Impact of Discounts**: Moderate discounts (20%-50%) consistently drive higher total sales compared to low or excessively high discounts, as revealed by both hypothesis testing and visualizations.
2. **Key Sales Drivers**: Features such as product ratings, actual price, and rating count significantly influence total sales, as highlighted in the feature importance analysis.
3. **Correlation Insights**: While discounts show a moderate positive correlation with total sales, excessively high discounts (>50%) appear to negatively impact revenue.
4. **Predictive Model Performance**: The Random Forest model demonstrated strong predictive capability with an R² score of 0.90, offering reliable sales predictions based on product features.

**Recommendations**: 
- Focus on optimizing discounts within the 20%-50% range for top-rated products.
- Increase marketing efforts for products with medium-to-high reviews to maximize revenue potential.
""")

st.markdown(
    """
    <hr style='border: 1px solid #e0e0e0;'>
    <footer class='footer'>
        <p>Amazon Sales Analysis App - Created by Zainab Junaid</p>
    </footer>
    """,
    unsafe_allow_html=True
)
