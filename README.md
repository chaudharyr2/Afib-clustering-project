# Afib-clustering-project
Afib doac bleeding prediction clustering exploration
Brief Introduction: 

The incidence and prevalence of atrial fibrillation (AF) are growing globally. The prevalence of AF has tripled over the last 50 years, affecting 46.3 million individuals globally in 20161,2. In people over 80, AF is a direct cause of 1 in 4 strokes3. Guidelines recommend the CHA₂DS₂-VASc score for stroke risk prediction, suggesting oral anticoagulant prescription for scores ≥2 in men and ≥3 in women4. Studies show that 70% of AF patients are on oral anticoagulants, 92% of whom are on direct oral anticoagulants (DOACs)5. However, DOAC use is linked to annual major bleeding rates of 2-4%, with case fatality rates of 8-15%6.   Algorithms have been developed to predict major bleeding risk among patients on oral anticoagulants, but their predictive yield has only been modest, with a c-statistic ranging from 0.58-0.61, a sensitivity of 60%, a specificity of 55%, and a positive predictive value of 3%. Most algorithms, developed for warfarin cohorts rather than DOACs, have overlapping risk factors for thromboembolism and bleeding, which limits their clinical utility7,8. Currently, bleeding prediction models are not recommended for guiding therapy. 

Incorporating clustering analysis into our research strategy offers a novel approach to dissecting the heterogeneity within the AF patient population on DOACs. By identifying distinct patient subgroups through clustering, we aim to unveil patterns that contribute to bleeding risk, thereby informing personalized treatment strategies and potentially identifying candidates for alternative therapies such as left atrial appendage occluder (LAAO) implantation. This study introduces a pioneering approach through clustering analysis to dissect the heterogeneity of bleeding risks in AF patients on DOACs, aiming to refine bleeding risk predictions beyond traditional models.

We hypothesize that an electronic health record (EHR) based machine learning model will identify characteristics of a high-risk bleeding cohort.

Research strategy: 

Aim. Build and compare ML models to predict major bleeding risk in patients with atrial fibrillation on DOACs using structured EHR data.

Hypothesis. Training ML models with EHR-derived variables will enhance the prediction of major bleeding incidents in DOAC-treated patients by 10-15%, improving risk reclassification compared to the traditional HAS-BLED score11.

Methods. Utilizing data (n~35,000) from the Atrial Fibrillation registry at UPMC, our methodology spans comprehensive data cleaning, imputation, and innovative feature engineering, followed by the application of diverse clustering algorithms to identify unique patient subgroups. This multifaceted approach is designed to enhance our understanding and prediction of bleeding risks. The data will analyze both categorical and continuous clinical data to predict the 1-year risk of major bleeding (ICH or GI bleed). We will use data at the time the patients make contact with the UPMC center of Atrial Fibrillation and a follow-up period of 1-year since the contact for the clinical outcome of major bleeding. All the data that are available to the clinician from the EHR at the time of first contact will be used for analysis for development of a pragmatic algorithm to predict bleeding risks. 

The methodology begins with data acquisition, followed by data cleaning, which includes ensuring unique values and one-hot encoding for categorical variables, and visualization and application of constraints for continuous variables. Subsequent steps address the percentage of missing values through determination of missingness at random, employing Little’s MCAR test and visualization techniques. Columns and rows with more than 60% missing values will be dropped if determined to be missing at random. Remaining missing values will be imputed with the median for continuous variables and mode for categorical variables.

Feature engineering, bolstered by domain expertise, will guide the selection or exclusion of variables from the prediction models. This involves assessing multicollinearity and variance inflation factors (VIF) and addressing variables with high correlations (>0.8) or VIF (>10) by excluding or carefully selecting among them.

Risk scoring systems will be calculated, generating summary statistics and p-values to assess variable significance in relation to major bleeding events, aiding hypothesis generation and further research. The dataset will then undergo stratified splitting into 70% training and two 15% test sets—one comprising non-comorbid individuals and the other a random sample. Class imbalance will be addressed by applying SMOTE (Synthetic Minority Over-sampling Technique), with parallel analysis conducted with and without SMOTE to evaluate performance differences.

Summary tables for the training and test datasets will provide insights into the demographics of the groups involved. Various clustering algorithms will be trained, incorporating 10-fold stratified cross-validation, and performance metrics (accuracy, precision, recall, F1 score, sensitivity, specificity, PPV, NPV, confusion matrix, AUC, and AU-PRC) will be reported for both training and stratified test sets, with each cross-validation fold detailed.

In exploring patient subgroups, I will employ a variety of clustering algorithms, each chosen for its unique methodological strengths and potential to uncover novel insights:

1.	K-Means Clustering: Chosen for its simplicity and effectiveness in large datasets, K-Means can help identify clear, separable subgroups based on clinical variables. Anticipated insights include the identification of distinct risk profiles that could inform targeted interventions.

2.	Hierarchical Clustering: This method's ability to create a dendrogram provides a visual representation of patient similarity, which can be invaluable for understanding the hierarchical structure of bleeding risks. It may reveal nested subgroups within high-risk categories, offering a nuanced view of patient stratification.

3.	DBSCAN: Particularly suitable for datasets with outliers and noise, DBSCAN's flexibility in identifying clusters of arbitrary shapes makes it an excellent tool for uncovering unusual patterns of risk factors that might be obscured in more traditional analyses.

4.	Gaussian Mixture Models (GMM): By assuming data is generated from several Gaussian distributions, GMM can identify overlapping subpopulations within the data. This approach is expected to reveal complex, multidimensional risk profiles, enhancing our understanding of bleeding risk heterogeneity.

5.	Spectral Clustering: Its use of similarity matrix eigenvalues to perform dimensionality reduction before clustering makes Spectral Clustering adept at identifying non-convex clusters. This algorithm is anticipated to uncover hidden patient groupings based on non-linear relationships among clinical features.

This diverse array of clustering methodologies is strategically selected to dissect the multidimensional nature of bleeding risk, enabling the derivation of both broad and nuanced insights into risk stratification. 

The study will evaluate traditional scoring systems (HASBLED, ATRIA, and ORBIT) against clinically established high-risk cut-points (3, 4, and 5, respectively). ML model performance and traditional scoring systems will be compared using AUC analysis. Descriptive analysis of identified clusters will be conducted, focusing on comorbidity prevalence, lab differences, and comparing clusters with the highest versus lowest prevalence of bleeding events to discern high-risk features. The nuanced understanding gained from these clustering analyses is anticipated to inform more personalized and effective risk mitigation strategies for patients on DOACs.

Of note, regarding the feasibility of the project, I already have the data which is preprocessed to the point of stratified train-test split. This semester’s focus will be on applying the nuances of clustering algorithms.

Table Shell 1: Comparison of ML Models Using Structured EHR Data

Model	C-statistic	Sensitivity (%)	Specificity (%)	Positive Predictive Value (%)	Negative predictive value (%)	F-1 score (%)

HAS-BLED Score	-	-	-	-		

K-Means clustering	-	-	-	-		

Hierarchial clustering	-	-	-	-		

DBSCAN	-	-	-	-		

Purpose: This table will compare the performance metrics of the ML models (and the traditionally used HAS-BLED score) in predicting major bleeding risk in patients with atrial fibrillation on DOACs using structured EHR data. 

 Dummy Figure 1: ROC Curve Comparing Models

A Receiver Operating Characteristic (ROC) curve plot comparing the ML models and the HAS-BLED score – below is an example of how it might look like

 

Purpose: This plot will allow for visual comparison of model performance in terms of sensitivity and specificity.



References

1.	Schnabel RB, Yin X, Gona P, et al. 50 year trends in atrial fibrillation prevalence, incidence, risk factors, and mortality in the Framingham Heart Study: a cohort study. Lancet. 2015;386(9989):154-162.

2.	Benjamin EJ, Muntner P, Alonso A, et al. Heart Disease and Stroke Statistics-2019 Update: A Report From the American Heart Association. Circulation. 2019;139(10):e56-e528.

3.	Wolf PA, Abbott RD, Kannel WB. Atrial fibrillation as an independent risk factor for stroke: the Framingham Study. Stroke. 1991;22(8):983-988.

4.	Writing Group M, January CT, Wann LS, et al. 2019 AHA/ACC/HRS focused update of the 2014 AHA/ACC/HRS guideline for the management of patients with atrial fibrillation: A Report of the American College of Cardiology/American Heart Association Task Force on Clinical Practice Guidelines and the Heart Rhythm Society. Heart Rhythm. 2019;16(8):e66-e93.

5.	Lund J, Saunders CL, Edwards D, Mant J. Anticoagulation trends in adults aged 65 years and over with atrial fibrillation: a cohort study. Open Heart. 2021;8(2).

6.	Siegal DM. What we have learned about direct oral anticoagulant reversal. Hematology Am Soc Hematol Educ Program. 2019;2019(1):198-203.

7.	Edmiston MK, Lewis WR. Bleeding Risk Scores in Atrial Fibrillation: Helpful or Harmful? J Am Heart Assoc. 2018;7(18):e010582.

8.	Yao X, Gersh BJ, Sangaralingham LR, et al. Comparison of the CHA(2)DS(2)-VASc, CHADS(2), HAS-BLED, ORBIT, and ATRIA Risk Scores in Predicting Non-Vitamin K Antagonist Oral Anticoagulants-Associated Bleeding in Patients With Atrial Fibrillation. Am J Cardiol. 2017;120(9):1549-1556.

9.	Osmancik P, Herman D, Neuzil P, et al. 4-Year Outcomes After Left Atrial Appendage Closure Versus Nonwarfarin Oral Anticoagulation for Atrial Fibrillation. J Am Coll Cardiol. 2022;79(1):1-14.

10.	Herrin J, Abraham NS, Yao X, et al. Comparative Effectiveness of Machine Learning Approaches for Predicting Gastrointestinal Bleeds in Patients Receiving Antithrombotic Treatment. JAMA Netw Open. 2021;4(5):e2110703.

11.	Petrazzini BO, Chaudhary K, Marquez-Luna C, et al. Coronary Risk Estimation Based on Clinical Data in Electronic Health Records. J Am Coll Cardiol. 2022;79(12):1155-1166.







