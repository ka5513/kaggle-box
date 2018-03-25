import pandas as pd

pd.set_option('display.width', 1000)


def prepare_data():
    train = pd.read_csv('../data/titanic/train.csv')
    test = pd.read_csv('../data/titanic/test.csv')

    train = process_dataset(train)
    test = process_dataset(test)

    train.to_csv('../data/titanic/train_processed.csv', index=False)
    test.to_csv('../data/titanic/test_processed.csv', index=False)


def extract_feature_from_name(df):
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}

    def map_title(title):
        mapping = title_mapping.get(title)
        return 0 if mapping is None else mapping

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df.Title = df.Title.replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
    df.Title = df.Title.map(map_title)

    return df


def normalize(series, decimals=5):
    return ((series - series.mean()) / series.std()).round(decimals)


def process_dataset(df):
    df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

    df.Embarked.replace(['C', 'Q', 'S'], [0, 1, 2], inplace=True)
    df.Sex.replace(['male', 'female'], [0, 1], inplace=True)

    df.Age.fillna(df.Age.mean(), inplace=True)
    df.Fare.fillna(df.Fare.mean(), inplace=True)
    df.Embarked.fillna(df.Embarked.value_counts().max(), inplace=True)

    df['FamilySize'] = df.SibSp + df.Parch
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    df = extract_feature_from_name(df)
    df.drop(['Name'], axis=1, inplace=True)

    df.Fare = normalize(df.Fare)
    df.Age = normalize(df.Age)

    print(df.head())
    return df


prepare_data()
