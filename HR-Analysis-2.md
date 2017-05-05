
# HR Analysis


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
emp = pd.read_csv('employees.csv')
```


```python
emp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>dept</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



We will classify the best employees or **Top Performers** as those with evaluations that are 1 standard deviation above the mean evaluation. We will compare them against everyone else.

We could also say that the "best" employees are those with an evaluation in the top quartile (0.87 evaluation and up) but let's stick into having just two classifications for a clearer analysis.

The analysis will focus on 4 points:
- Top Performers who left the company
- Top Performers who stayed with the company
- Lower Performers who left the company
- Lower Performers who stayed with the company

## Finding the good performers


```python
avg_evaluation = emp['last_evaluation'].mean()
std_evaluation = emp['last_evaluation'].std()
emp['std_performance'] = (emp['last_evaluation'] - avg_evaluation) / std_evaluation
emp['performance_differential'] = emp['last_evaluation']-(avg_evaluation+std_evaluation)
#emp
```


```python
def performance_classification(row):
    if row['performance_differential']>=0:
        performance_class = 'Top Performer'
    else:
        performance_class = 'Lower Performer'
    return (performance_class)
emp['classification'] = emp.apply(performance_classification, axis=1)
```

## Calculating Average Daily Hours as a Better Indicator work-life balance
### *Assuming an average of 22 working days per month*


```python
emp['daily_hours'] = emp['average_montly_hours']/22
#emp
```

## Prepping for Pivot Table

**ie, isolating non-numeric values (such as performance classification AND left or stayed) for averaging numerical values across them**


```python
left_dict = {1: 'left', 0: 'stayed'}

emp['left(as_string)'] = (emp['left'].map(left_dict))
```

*keeping numeric columns only*


```python
# Salary is an important indicator but since it is not numeric we will have to map the three classes of salaries
# to arbitrary numbers as shown below

salary_num_dict = {'low':30000, 'medium':60000, 'high':90000}
emp['salary_num'] = (emp['salary'].map(salary_num_dict))

numeric_columns = emp._get_numeric_data()

new_emp = emp

new_emp['performance_group'] = new_emp['left(as_string)'] + ':' + new_emp['classification']

num_pivot = new_emp.pivot_table(index='performance_group',values = numeric_columns, aggfunc='mean')
num_pivot = num_pivot.transpose()

num_pivot
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>performance_group</th>
      <th>left:Lower Performer</th>
      <th>left:Top Performer</th>
      <th>stayed:Lower Performer</th>
      <th>stayed:Top Performer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Work_accident</th>
      <td>0.050988</td>
      <td>0.038425</td>
      <td>0.175504</td>
      <td>0.172983</td>
    </tr>
    <tr>
      <th>average_montly_hours</th>
      <td>187.898024</td>
      <td>254.862632</td>
      <td>198.254872</td>
      <td>202.358003</td>
    </tr>
    <tr>
      <th>daily_hours</th>
      <td>8.540819</td>
      <td>11.584665</td>
      <td>9.011585</td>
      <td>9.198091</td>
    </tr>
    <tr>
      <th>last_evaluation</th>
      <td>0.623660</td>
      <td>0.947666</td>
      <td>0.660180</td>
      <td>0.941899</td>
    </tr>
    <tr>
      <th>left</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>number_project</th>
      <td>3.310672</td>
      <td>5.179635</td>
      <td>3.777137</td>
      <td>3.825680</td>
    </tr>
    <tr>
      <th>performance_differential</th>
      <td>-0.263611</td>
      <td>0.060395</td>
      <td>-0.227091</td>
      <td>0.054628</td>
    </tr>
    <tr>
      <th>promotion_last_5years</th>
      <td>0.007115</td>
      <td>0.000961</td>
      <td>0.026347</td>
      <td>0.025858</td>
    </tr>
    <tr>
      <th>salary_num</th>
      <td>42557.312253</td>
      <td>42161.383285</td>
      <td>49721.284703</td>
      <td>48738.296924</td>
    </tr>
    <tr>
      <th>satisfaction_level</th>
      <td>0.399787</td>
      <td>0.538069</td>
      <td>0.663585</td>
      <td>0.680013</td>
    </tr>
    <tr>
      <th>std_performance</th>
      <td>-0.540060</td>
      <td>1.352837</td>
      <td>-0.326707</td>
      <td>1.319149</td>
    </tr>
    <tr>
      <th>time_spend_company</th>
      <td>3.530830</td>
      <td>4.716619</td>
      <td>3.379858</td>
      <td>3.380740</td>
    </tr>
  </tbody>
</table>
</div>



The first thing to notice is that top performers who left had on average done more projects than any of the 3 other groups of employees **AND** have worked 52 more hours than those who stayed. 

My initial thoughts indicate that Top Performers who left were overworked, by taking on more project and working longer hours, but stayed longer at the company in the hopes of a promotion that never came. On average, their salaries were not the highest indicating that they were **underappreciated** by their managers.

Also, top performers stayed, on average, for a longer period of time at the company than the other 3 groups.

Top performers who stayed or **haven't left yet** are almost 10 times more likely to receive a promotion.

## Who works most?


```python
plt.figure(figsize=(8,6))
sns.boxplot(x='performance_group',y='average_montly_hours',data=new_emp)
plt.title('Box Plot of Average Monthly Hours Worked by Performance Group')
plt.show()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-31-38b09f3d0c2c> in <module>()
          1 plt.figure(figsize=(8,6))
          2 sns.boxplot(x='performance_group',y='average_montly_hours',data=new_emp)
    ----> 3 plt.title('Box Plot of Average Monthly Hours Worked by Performance Group')
          4 plt.show()


    TypeError: 'str' object is not callable



![png](HR-Analysis-2_files/HR-Analysis-2_20_1.png)


The above boxplot illustrates how top performer employees worked much harder than the rest.


```python
plt.figure(figsize=(8,6))
sns.boxplot(x='performance_group',y='number_project',data=new_emp)
plt.title('Box Plot of Number of Projects Taken On by Performance Group')
plt.show()
```

Here also, the top performers who left the company were more likely to take on projects.

## How much were the employees who left making?


```python
plt.figure(figsize=(8,6))
sns.distplot(new_emp[new_emp['left']==1]['salary_num'],kde=False,bins=3,norm_hist=True)
plt.title('Salary Distribution of Employees who Left the Company')
plt.xlim([30000,90000])
plt.show()
```

Most employees who left the company belonged to the lowest tranch of the income bracket.

## When are the top perfomers leaving the company?


```python
top_left = emp[(emp['performance_group']=="left:Top Performer")]              

top_left['time_spend_company'].value_counts(normalize=True)
```


```python
plt.figure(figsize=(8,6))
sns.distplot(top_left['time_spend_company'],kde=False)
plt.title('Distribution of Time Spent at Company Among the Top Performers Who Have Already Left')
plt.ylabel('Employee Count')
plt.show()
```

## Correlations for the top performers


```python
#new_emp
```


```python
columns = ['left','average_montly_hours', 'number_project','time_spend_company', 'promotion_last_5years']

top_emp = new_emp[new_emp['classification']=='Top Performer']
not_top_emp = new_emp[new_emp['classification']!='Top Performer']
```

## Top Performers Correlation Matrix


```python
top_corr = top_emp.corr().loc[columns].transpose()
top_corr
```

## Lower Performers Correlation Matrix


```python
not_top_corr = not_top_emp.corr().loc[columns].transpose()
not_top_corr
```

## Initial Insights from the Correlation Tables

### For the top performers:

- Leaving the company is correlated with: hours worked, number of projects taken, and time spent at the company
- Number of projects taken is negatively correlated with satisfaction levels
- There is no correlation between receiving a promotion in the last 5 years and working longer, or taking projects

### For the lower performers:

- Leaving the company is highly correlated with dissatisfaction levels of the employees
- As expected, average hours worked is correlated with number of projects taken on
- Also, as expected, receiving a promotion did not correlate with any of other metrics, which makes sense since we do not expect the lower performers to receive promotions

## How satisfied were the top performers who left the company?


```python
plt.figure(figsize=(8,6))
sns.distplot(top_left['satisfaction_level'],kde=False,bins=40)
plt.title('Satisfaction Level Distribution Among Top Employees Who Left The Company')
plt.ylabel('Employee Count')
plt.show()
```

## Conclusions

**Top performers** seem to be working longer hours with little appreciation. They're being assigned on more projects than the others and they're not happy about it, hence the high number of dissatisfied top performers (as shown in the figure above). Almost half of the **top performers** leave the company after 5 years. They are overworked and are unlikely to receive a promotion.

## Taking a dive into departmental stats


```python
total_vc = new_emp['dept'].value_counts()
left_vc = new_emp[new_emp['left']==1]['dept'].value_counts()
#sns.countplot(x='dept',data=total_vc)
dept_df = pd.DataFrame(total_vc)
left_dept_df = pd.DataFrame(left_vc)
combined_df_dept = pd.concat([dept_df,left_dept_df],axis=1)
combined_df_dept.columns = ['total', 'left']
combined_df_dept['percentage_left'] = combined_df_dept['left'] / combined_df_dept['total']
combined_df_dept.sort('percentage_left',ascending=False)
```

The departments with the highest employee turnover are HR followed by Accounting. Even though the Sales department has the highest number of employees who left, they form a smaller fraction of the total number of employees in the department.

R&D and Management are the least likely to churn employees.

**Visualizing the table above in a bar plot:**


```python
plt.figure(figsize=(8,6))
new_emp['dept'].value_counts().plot(kind='bar')
new_emp[new_emp['left']==1]['dept'].value_counts().plot(kind='bar', color = 'red')
plt.title =('Share of Employees who Left by Department')
plt.xlabel('Departments')
plt.ylabel('Employee Count')
plt.show()
```


```python
new_emp_by_dept = new_emp.groupby('dept').mean()
#new_emp_by_dept
```

### Further Analysis On A Departmental Level


```python
for col in new_emp_by_dept.columns:
    new_emp_by_dept_sorted = new_emp_by_dept.sort(col, ascending = False)
    new_emp_by_dept_sorted[col].plot(kind = 'bar')
    plt.ylabel(col)
    plt.xticks( rotation = 30)
    plt.show()
```

### Main Takeaways from Departmental Analysis

- Management have the **highest** satisfaction level, evaluation level, rate of promotion, and the **lowest** turnover rate. They are the third most overworked department in terms of monthly hours and the second highest in number of projects taken on. Promotion for managers usually result in a big boost in pay (salary + stock options), so it is obvious why they wouldn't leave the company.
- The Technical department seems to be the most overworking as they have the **highest** average monthly hours worked and number of projects undertaken, but they are amont the **least satisfied** employees even though their average evaluation is among the top, which would explain the high rate of technical employees leaving the company.
- Accounting and Support are amongst the most hardworking but they are among the least likely to get promoted or be satisfied, which explains the high rate of employee churn.
- It is also interesting to note that Marketing, R&D, and Sales are among the most likely to be promoted and least likely to leave, which could mean that management relies on these 3 departments more than the others, making the more hardworking people at the other departments upset with the current management style.


```python

```
