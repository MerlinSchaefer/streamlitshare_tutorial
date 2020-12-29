##imports
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

##read in data

housing = pd.read_csv("housing.csv")
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
### Displaying text
##The title of your App can be displayed with st.title
st.title("Housing Price Predictions") #you could also use st.write("# YOURTITLE")

### Displaying an image
## st.image allows you to display an image from a url or numpy array, a BytesIO object and more
st.image("https://storage.googleapis.com/kaggle-datasets-images/24824/31630/a5f5ce1e4b4066d1f222e79e8286f077/dataset-cover.jpg?t=2018-05-03-00-52-48",
         width = 700)

##st.write allows you to write text either with single quotes for one line in markdown format
st.write("Predicting the median value of a house within a district.")
##or with triple quotes for multiple lines
st.write("""  
         The houses are located in **California,USA**
         
         *Note*: This was create for a streamlit/streamlit-sharing tutorial,
         not all features of the original dataset are used for the prediction, the machine learning model is not good or tuned in any way 
         it just serves as an example for streamlit deployment.
         
         """)



###Displaying data
st.write("# Displaying Data")
##display dataframe
##add column selector
col_names = housing.columns.tolist()
st.dataframe(housing[st.multiselect("Columns to display",col_names, default = col_names)])


###Plotting
st.write("# Plotting")
##display figures
fig,ax = plt.subplots() #must create a subplot
ax = sns.countplot(housing["ocean_proximity"], palette ="tab20")
sns.despine()
st.pyplot(fig)

st.write("## You could showcase geographical data with seaborn or matplotlib")
fig2,ax2 = plt.subplots()
ax2 = sns.scatterplot(x = housing["longitude"], y = housing["latitude"])
sns.despine()
st.pyplot(fig2)


##display map data
st.write("## But Streamlit makes plotting geographical data much easier and better")
st.map(housing, zoom = 4.5)

###Machine Learning
st.write("# Machine Learning App")

ml_model = pickle.load(open("model.pkl", "rb")) #load the model

##create the sidebar
st.sidebar.header("User Input Parameters")

##create function for User input
def get_user_input():
    housing_median_age = st.sidebar.slider("How old is the median house in the district?",
                          housing["housing_median_age"].min(),
                          housing["housing_median_age"].max(),
                          housing["housing_median_age"].mean())
    rooms_per_household = st.sidebar.slider("How many rooms per household are typically found in the district",
                          housing["rooms_per_household"].min(),
                          housing["rooms_per_household"].max(),
                          housing["rooms_per_household"].mean())
    households = st.sidebar.slider("How many households are in the district?",
                          housing["households"].min(),
                          housing["households"].max(),
                          housing["households"].mean())
    median_income = st.sidebar.slider("What is the median income in the district?",
                           housing["median_income"].min(),
                           housing["median_income"].max(),
                           housing["median_income"].mean())
    features = pd.DataFrame({"housing_median_age":housing_median_age,
                             "rooms_per_household":rooms_per_household,
                             "households":households,
                             "median_income":median_income}, index = [0])
    return features


input_df = get_user_input() #get user input from sidebar

prediction = ml_model.predict(input_df) #get predicitions
#display predictions
st.subheader("Prediction")
st.write("**The median price of a house in the district is: $**",str(round(prediction[0],2)))

