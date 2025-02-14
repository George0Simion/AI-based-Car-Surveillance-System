
-> librarie folosita pentru manipularea datelor, analizarea lor, si manageruirea dataseturilor eficient
-> facuta deasupra la numpy

#### 1. Data structures

-> foloseste serii: *__Series__*
-> seriile implementate folosind vectori numpy care indexeaza metadata ul

#### 2. DataFrame: 2D Data

-> un tabel 
-> atribute:
* .shape()
* .columns()
* .index()
* .dtypes()



		Panda poate da load si save (si sa scrie) unor dataseturi in diferite formate
	ex: csv, excel, json, sql

#### 3. Data Manipulation

-> poti sa selectezi coloane, linii si sa adaugi / remove coloane:
df["Age] -> selecteaza coloana(si practic intoarce o serie)
df.loc[0] -> selecteaza linia dupa index
df.iloc[1] -> selecteaza dupa numarul liniei
df["Bonus"] = df["Salary"] * 0.1  # New column
df.drop("Bonus", axis=1, inplace=True)  # Remove column
df.sort_values(by="Age", ascending=False, inplace=True)
df.groupby("Age")["Salary"].mean()


#### 4. Handling Missing Data

-> poti vedea valori care lipsesc, mai apoi putand sa le dai fill sau drop:
df.isnull().sum()
df.fillna(value={"Salary": df["Salary"].mean()}, inplace=True)  # Fill NaN with mean
df.dropna(inplace=True)

#### 5. Aggregation & Statistics
print(df.describe())  # Summary statistics
print(df['Salary'].mean())  # Average Salary
print(df.groupby('Age')['Salary'].sum())  # Sum of Salary by Age

#### 6. Merging & Joining DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Salary': [50000, 60000, 70000]})

-> Inner Join (Default)
merged = pd.merge(df1, df2, on='ID')

-> Left Join
left_join = pd.merge(df1, df2, on='ID', how='left')

-> Outer Join
outer_join = pd.merge(df1, df2, on='ID', how='outer')

#### 10. Summary
| **Concept**     | **Explanation**                                      |
|----------------|------------------------------------------------------|
| **Series**     | 1D labeled array (column of data)                    |
| **DataFrame**  | 2D labeled table (spreadsheet-like)                  |
| **Selection**  | `df.loc[]`, `df.iloc[]`, column selection             |
| **Filtering**  | `df[df['Age'] > 25]`                                  |
| **Missing Data** | `dropna()`, `fillna()`                              |
| **Aggregation** | `groupby()`, `mean()`, `sum()`                       |
| **Merging**    | `merge()` for joins                                   |
| **Performance** | `chunksize`, `astype()` for memory optimization      |
