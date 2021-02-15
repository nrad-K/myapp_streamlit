import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import time

def main():
    #model learning 
    dataset = pd.read_csv('./data/preprocessing_data.csv', index_col=0)


    X_pos = dataset['POSIX'].values
    X = dataset[['POSIX', 'year', 'month', 'dayofweek', '1dayago', '2dayago', '3dayago', '4dayago', '5dayago', '6dayago', '7dayago']].values
    y = dataset['平均気温(℃)'].values

    N = len(X)
    N_train = round(N*0.8)

    X_pos = X_pos.reshape(-1, 1)
    X = X
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = X[7:N_train], X[N_train:], y[7:N_train], y[N_train:]
    X_pos_train, X_pos_test = X_pos[7:N_train], X_pos[N_train:]

    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    #plot
    fig = plt.figure(figsize=(28, 18))
    plt.plot(X_pos, y, color='gray', linestyle='dashdot', label='true_data')
    plt.plot(X_pos_train, y_train, '+', label='train_data')
    plt.plot(X_pos_test, y_test, '--', label='test_data')
    plt.plot(X_pos_test, y_pred, '*', label='pred')
    plt.xlabel('POSIX')
    plt.ylabel('temperature')
    plt.grid()
    plt.legend()



    train_score_text = "Train-score: {:.2f}".format(model.score(X_train, y_train))
    test_score_text = "Test-score: {:.2f}".format(model.score(X_test, y_test))


    #application
    st.title('気象庁 平均気温(2000/01/01~2021/02/13) データ分析')

    st.header('気象庁平均気温データ')
    st.dataframe(dataset, width=800, height=300)

    st.header('データのグラフ')
    st.pyplot(fig)

    st.header('モデル(RandomForestRegressor)のスコア')
    st.text(train_score_text)
    st.text(test_score_text)


    #sidebar
    st.sidebar.header('平均気温を予測')

    input_datetime = st.sidebar.date_input("予測したい年月日を入力してください:",)
    input_data = [[input_datetime]]

    for i in range(7):
        tmp_text = '{}日前の気温を入力してください:'.format(i+1)
        input_dayago = float(st.sidebar.number_input(tmp_text, 0.0))
        data = input_data[0].append(input_dayago)


    #data preprocessing   
    data_df = pd.DataFrame(input_data, columns=['datetime', '1dayago', '2dayago', '3dayago', '4dayago', '5dayago', '6dayago', '7dayago'])

    data_df['datetime'] = pd.to_datetime(data_df['datetime'])
    data_df['1dayago'] = pd.to_numeric(data_df['1dayago'])
    data_df['2dayago'] = pd.to_numeric(data_df['2dayago'])
    data_df['3dayago'] = pd.to_numeric(data_df['3dayago'])
    data_df['4dayago'] = pd.to_numeric(data_df['4dayago'])
    data_df['5dayago'] = pd.to_numeric(data_df['5dayago'])
    data_df['6dayago'] = pd.to_numeric(data_df['6dayago'])
    data_df['7dayago'] = pd.to_numeric(data_df['7dayago'])
    data_df['POSIX'] = data_df['datetime'].astype(np.int64).values//10**9
    data_df['year'] = data_df['datetime'].dt.year
    data_df['month'] = data_df['datetime'].dt.month
    data_df['dayofweek'] = data_df['datetime'].dt.dayofweek

    input_X = data_df[['POSIX', 'year', 'month', 'dayofweek', '1dayago', '2dayago', '3dayago', '4dayago', '5dayago', '6dayago', '7dayago']].values



    st.header('予測した平均気温')
    if st.sidebar.button('予測を開始!'):


        my_bar = st.progress(0)
        iteration = st.empty()
        for percent_complete in range(100):
            iteration.text(f'学習中{percent_complete+1}%')
            my_bar.progress(percent_complete + 1)
            time.sleep(0.1)
        
        pred = model.predict(input_X)
        st.text('学習完了!')
        st.write(f'{input_datetime}の平均気温は',pred[0],'℃です！')
    else:
        st.write('ここに予測された値が表示されます')




if __name__ == '__main__':
    main()