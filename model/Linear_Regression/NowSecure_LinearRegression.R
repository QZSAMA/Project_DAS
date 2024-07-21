library(ggplot2)
library(reshape)

vulnerabilities<-read.csv('C:/Users/atuwh/Documents/YL/Gatech/INTA6450/NowSecure/Project_DAS/data/dataset_cleaned - removedOutliers.csv')
#removed 5 outliers

summary(vulnerabilities)
# Print summary statistics for the data frame

table(vulnerabilities$Malicious.URL.count)
# Tabulate Malicious.URL.count levels in the sample

ggplot(data=vulnerabilities, aes(x=Malicious.URL.count, y=score)) + geom_point() + stat_smooth(formula=y~x)
# Look at the distribution of Malicious.URL.count and score in a scatter plot

model.results <- lm(score ~ Malicious.URL.count, data=vulnerabilities)
# Fit a linear model where y is score and x is Malicious.URL.count
print(model.results)
# print a simple version of the model results
summary(model.results)
# Print a more detailed version of model results

vulnerabilities<-read.csv('C:/Users/atuwh/Documents/YL/Gatech/INTA6450/NowSecure/Project_DAS/data/dataset_cleaned.csv')
#cleaned raw dataset (with the 5 outliers)

ggplot(data=vulnerabilities, aes(x=android_target_sdk_min, y=score)) + geom_point() + stat_smooth(formula=y~x)
# Look at the distribution of android_target_sdk_min and score in a scatter plot

ggplot(data=vulnerabilities, aes(x=android_target_sdk_min, y=score)) + geom_violin(aes(group=android_target_sdk_min)) + stat_smooth(data=vulnerabilities, aes(x=android_target_sdk_min, y=score), method = 'lm')
# A picture of how the linear model does, with marginal distributions
# Save the picture by uncommenting the line below:
# ggsave('violin.png', width=7, height=5, units = "in")


ggplot(data=vulnerabilities, aes(x = as.factor(android_target_sdk_min), y = score)) + geom_boxplot() +
  labs(title = "Box Plot of Scores by Android Target SDK Min",
       x = "Android Target SDK Min (TRUE/FALSE)",
       y = "Score") +
  theme_minimal()
# Create a box plot for Scores and Android Target SDK Min

ggplot(data=vulnerabilities, aes(x=api_resource_misconfiguration, y=score)) + geom_point() + stat_smooth(formula=y~x)
# Look at the distribution of api_resource_misconfiguration and score in a scatter plot

ggplot(data=vulnerabilities, aes(x=api_resource_misconfiguration, y=score)) + geom_violin(aes(group=api_resource_misconfiguration)) + stat_smooth(data=vulnerabilities, aes(x=api_resource_misconfiguration, y=score), method = 'lm')
# A picture of how the linear model does, with marginal distributions
# Save the picture by uncommenting the line below:
# ggsave('violin.png', width=7, height=5, units = "in")

ggplot(data=vulnerabilities, aes(x = as.factor(api_resource_misconfiguration), y = score)) + geom_boxplot() +
  labs(title = "Box Plot of Scores by Api Resource Misconfiguration",
       x = "Api Resource Misconfiguration(TRUE/FALSE)",
       y = "Score") +
  theme_minimal()
# Create a box plot for Scores and Api Resource Misconfiguration

ggplot(data=vulnerabilities, aes(x=apk_leaked_data_sdcard, y=score)) + geom_point() + stat_smooth(formula=y~x)
# Look at the distribution of apk_leaked_data_sdcard and score in a scatter plot

ggplot(data=vulnerabilities, aes(x=apk_leaked_data_sdcard, y=score)) + geom_violin(aes(group=apk_leaked_data_sdcard)) + stat_smooth(data=vulnerabilities, aes(x=apk_leaked_data_sdcard, y=score), method = 'lm')
# A picture of how the linear model does, with marginal distributions
# Save the picture by uncommenting the line below:
# ggsave('violin.png', width=7, height=5, units = "in")

ggplot(data=vulnerabilities, aes(x = as.factor(apk_leaked_data_sdcard), y = score)) + geom_boxplot() +
  labs(title = "Box Plot of Scores by Apk Leaked Data Sdcard",
       x = "Apk Leaked Data Sdcard(TRUE/FALSE)",
       y = "Score") +
  theme_minimal()
# Create a box plot for Scores and Apk Leaked Data Sdcard


model.results.detail <- lm(score ~ android_target_sdk_min + api_resource_misconfiguration + apk_leaked_data_sdcard, data=vulnerabilities)
summary(model.results.detail)
