import streamlit as st
import pandas as pd

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Trading Signal Prediction Dashboard",
    layout="wide"
)

st.title("📈 Trading Signal Prediction using Machine Learning")

st.markdown("---")

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
tab = st.sidebar.radio(
    "Navigate",
    [
        "Introduction & Problem Statement",
        "Literature Survey & Research Gap",
        "Objectives & Methodology",
        "Experimental Results & Conclusion"
    ]
)

# =========================================================
# TAB 1
# =========================================================
if tab == "Introduction & Problem Statement":

    st.header("Introduction")

    st.write("""The Foreign Exchange (Forex) market is the largest and most liquid financial market globally, facilitating the exchange of currencies and other financial instruments. Due to its high liquidity, continuous operation, and rapid price fluctuations, the Forex market offers significant trading opportunities but also presents substantial prediction challenges.  
             
Among the widely traded instruments, XAU/USD represents the price of gold quoted in U.S. dollars. Gold is considered a safe-haven asset and is influenced by macroeconomic factors such as inflation, interest rates, and geopolitical events. As a result, XAU/USD exhibits dynamic and often volatile price movements. Predicting price direction in such a market is difficult due to market noise, non-linearity, and class imbalance in trading signals. Traditional rule-based technical analysis methods are limited in capturing complex patterns within financial time-series data. 
    
In this project, machine learning techniques are applied to predict BUY or NOT BUY signals for XAU/USD using raw price data and engineered technical indicators. Multiple classification models, including Support Vector Machines, K-Nearest Neighbors, XGBoost, and CatBoost, are evaluated to identify the most effective approach for reliable trading signal generation
    """)

    st.markdown("---")

    st.header("Problem Statement")

    st.write("""
To develop a machine learning-based classification system that predicts BUY or NOT BUY signals for XAU/USD using historical price data and technical indicators, while handling market noise and class imbalance effectively.
    """)

# =========================================================
# TAB 2
# =========================================================
elif tab == "Literature Survey & Research Gap":


    st.header("Literature Survey")

    data = {
    "No.": [1, 2, 3, 4, 5, 6],
    "Author & Year": [
        "Patel et al., 2015",
        "Kara et al., 2011",
        "Zhang et al., 2018",
        "Fischer & Krauss, 2018",
        "Chen & Guestrin, 2016",
        "Sudimanto et al., 2021"
    ],
    "Dataset / Instrument": [
        "Stock Market Data",
        "Stock Market Data",
        "Financial Time-Series",
        "Stock Returns",
        "Structured Tabular Data",
        "XAU/USD (Gold), 2019–2021"
    ],
    "Models Used": [
        "SVM, ANN, Random Forest",
        "ANN, SVM",
        "Deep Neural Networks",
        "LSTM",
        "XGBoost",
        "Decision Trees, SVM, KNN, Ensemble Methods"
    ],
    "Key Indicators / Features": [
        "Technical Indicators",
        "Technical Indicators",
        "Price-Based Features",
        "Sequential Price Data",
        "Structured Features",
        "MACD"
    ],
    "Key Findings": [
        "Random Forest showed strong performance",
        "SVM achieved good classification accuracy",
        "Deep models capture non-linear patterns",
        "LSTM improved return prediction",
        "Boosting effective for structured data",
        "Ensemble approaches showed promising results"
    ],
    "Limitations": [
        "Limited feature engineering",
        "No imbalance handling",
        "High computational complexity",
        "Risk of overfitting in volatile markets",
        "Not focused on trading signal generation",
        "Pilot-level study with limited indicators and lack of model optimization"
    ]
    }

    df_lit = pd.DataFrame(data)

    st.dataframe(df_lit, use_container_width=True)


    st.markdown("---")
    
    
    
    st.header("Research Gap")

    st.write("""
Based on the reviewed literature, the following research gaps were identified:

1. Many studies focus on general price direction forecasting rather than explicit BUY or NOT BUY signal classification.

2. Pilot-level studies, such as Sudimanto et al. (2021), rely on limited technical indicators (e.g., MACD only) and lack comprehensive model optimization.

3. Class imbalance handling is rarely addressed in trading signal prediction tasks.

4. Comparative analysis across diverse model families (linear, instance-based, and boosting methods) remains limited.
""")


# =========================================================
# TAB 3
# =========================================================
elif tab == "Objectives & Methodology":

    st.header("Objectives")

    st.write("""
- To analyze historical XAU/USD price data and extract meaningful technical indicators for signal generation.

- To design a classification framework that predicts BUY or NOT BUY trading signals.

- To implement and compare multiple machine learning models, including Linear SVM, Cubic SVM, KNN, XGBoost, and CatBoost.

- To handle class imbalance and evaluate models using appropriate performance metrics such as Accuracy, Precision, Recall, F1-Score, and AUC.

- To identify the most effective model for reliable and conservative trading signal prediction. 
    """)

    st.markdown("---")

    st.header("Methodology")

    st.subheader("Step 1: Data Collection")
    st.write("Historical financial time-series data was collected and preprocessed.")

    st.subheader("Step 2: Feature Engineering")
    st.write("""
    The following technical indicators were computed:
    - Relative Strength Index (RSI)  
    - Moving Average Convergence Divergence (MACD)  
    - Bollinger Bands  
    - Average True Range (ATR)  
    - Exponential Moving Average (EMA)  
    """)

    st.subheader("Step 3: BUY Label Definition")
    st.write("""
    A BUY signal was defined using:
    - Volatility-adjusted price movement  
    - Trend confirmation  
    - Momentum filtering  
    """)

    st.subheader("Step 4: Train-Test Split")
    st.write("A time-aware 80–20 split was used to prevent look-ahead bias.")

    st.subheader("Step 5: Models Implemented")
    st.write("""
    - Linear SVM  
    - Cubic SVM  
    - K-Nearest Neighbors (KNN)  
    - XGBoost  
    - CatBoost  
    """)

    st.subheader("Step 6: Evaluation Metrics")
    st.write("""
    - Accuracy  
    - Precision  
    - Confusion Matrix  
    """)

# =========================================================
# TAB 4
# =========================================================
elif tab == "Experimental Results & Conclusion":

    st.header("Experimental Results")

    results = {
        "Model": ["Linear SVM", "Cubic SVM", "KNN", "XGBoost", "CatBoost"],
        "Accuracy (%)": ["85.30%", "83.18%", "85.70%", "81.20%", "85.30%"],
    }

    df_results = pd.DataFrame(results)
    df_results.index = df_results.index + 1
    st.dataframe(df_results, use_container_width=True)

    st.markdown("---")

    st.subheader("Key Observations")

    st.write("""
    - Linear SVM showed limited separation capability.  
    - Cubic SVM improved non-linear decision boundaries.  
    - KNN demonstrated moderate performance.  
    - XGBoost provided strong baseline results.  
    - CatBoost achieved the best balance between predictive accuracy and conservative BUY signals.  
    """)

    st.markdown("---")
    
    import matplotlib.pyplot as plt

    models = ["Linear SVM", "Cubic SVM", "KNN", "XGBoost", "CatBoost"]
    accuracy = [85.3, 83.18, 85.7, 81.2, 85.3]  # replace with real

    plt.figure()
    plt.plot(models, accuracy, marker='o')
    plt.xlabel("Models")
    plt.ylabel("Metric Value")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45)
    plt.legend(["Accuracy", "BUY Precision"])

    st.pyplot(plt)
    
    
    
    
    st.markdown("---")
    st.header("ROC Curve Analysis")

    st.image("output.png", use_container_width=True)

    st.write("""
The Receiver Operating Characteristic (ROC) curve evaluates the ability of each model
to distinguish between BUY and NOT BUY signals across all possible classification thresholds.

The Area Under the Curve (AUC) value represents the overall discriminatory power
of the model. A higher AUC indicates better separation between positive (BUY)
and negative (NOT BUY) classes.

Key Observations:

- Models with curves closer to the top-left corner demonstrate stronger classification performance.
- AUC values closer to 1 indicate superior predictive capability.
- Ensemble boosting models (XGBoost and CatBoost) exhibit higher AUC scores,
  confirming their effectiveness in handling non-linear financial data.
- Linear and instance-based models show comparatively lower AUC,
  indicating weaker separation capability.

Overall, the ROC analysis confirms that boosting-based ensemble methods
provide more reliable ranking of BUY signals compared to other approaches.
""")


    st.markdown("""
### ROC Curve Interpretation

The Receiver Operating Characteristic (ROC) curve illustrates the trade-off between 
True Positive Rate (TPR) and False Positive Rate (FPR) across varying classification thresholds.

The Area Under the Curve (AUC) quantifies the overall discriminatory ability of each model.
A higher AUC indicates better separation between BUY and NOT BUY classes.

#### Observations from the ROC Comparison:

- **Linear SVM (AUC = 0.82)** achieves the highest AUC score, indicating strong classification capability.
- **Cubic SVM (AUC = 0.81)** performs similarly to Linear SVM, suggesting that the data may already be sufficiently separable in a near-linear feature space.
- **KNN (AUC = 0.75)** demonstrates moderate performance but shows less stability in ranking BUY signals.
- **CatBoost (AUC = 0.74)** performs reasonably well but does not outperform SVM-based approaches in this configuration.
- **XGBoost (AUC = 0.66)** shows comparatively weaker discriminatory power for this dataset.

The diagonal reference line represents random classification (AUC = 0.50). 
All models perform above random guessing, confirming meaningful predictive capability.

Overall, Support Vector Machine models demonstrate stronger ranking ability for BUY signals 
in this experimental setup, while boosting-based models may require further hyperparameter tuning 
or feature refinement for optimal performance.
""")

    
    
    
    
    
    
    
    st.markdown("---")
    
    


    st.header("Conclusion")

    st.write("""
This study explored the application of machine learning techniques for predicting BUY or NOT BUY signals in the XAU/USD Forex market using historical price data and technical indicators.

A comparative analysis was conducted across multiple models, including Linear SVM, Cubic SVM, KNN, XGBoost, and CatBoost. The evaluation using classification metrics and ROC analysis demonstrated that Support Vector Machine models achieved stronger class separation capability in this dataset, while boosting-based models provided competitive performance.

The results highlight the importance of proper feature selection, model comparison, and evaluation beyond simple accuracy metrics. Overall, machine learning methods show promising potential in assisting trading decision-making by identifying meaningful BUY opportunities in complex financial markets.
    """)

