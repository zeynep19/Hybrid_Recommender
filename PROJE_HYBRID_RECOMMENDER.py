import pandas as pd

pd.set_option('display.max_columns', 20)

#########################################################################
# Görev-1 Veri Hazırlama işlemlerini gerçekleştiriniz.
#########################################################################
movie = pd.read_csv('pythonProject/datasets/movie.csv')
rating = pd.read_csv('pythonProject/datasets/rating.csv')

def create_user_movie_df():
    # movie ve rating veri setlerini movieId sütunu bazında birleştirdik.
    df = movie.merge(rating, how="left", on="movieId")
    # Filmlere yapılan yorum sayılarını comment_counts'a atadık.
    comment_counts = pd.DataFrame(df["title"].value_counts())
    # 1000 den az sayıda yorum alan filmleri rare_movies'e atadık.
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    # rare_movies olan filmleri çıkardık.
    common_movies = df[~df["title"].isin(rare_movies)]
    # Satırlarda userId, sütunlarda title yani filmin ismi, kesişimlerinde yani değerler kısmında ise oylama puanının
    # yer aldığı matrisi oluşturduk.
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# user_movie_df matrisinden rastgele bir kullanıcı seçtik.(userId=90740)
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=8).values)

#########################################################################
# Görev-2 Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.
#########################################################################
# random_user kullanıcısının id'sine ait satırı random_user_df'e atadık.
random_user_df = user_movie_df[user_movie_df.index == random_user]

# random_user_df'deki filmlerin boş olmayanlarını yani kullanıcının izlediklerini movies_watched'a atadık.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Kullanıcının izlediği filmlerin sayısını çıkardık.(70)
len(movies_watched)

#########################################################################
# Görev-3 Aynı filmleri izleyen diğer kullanıcıların verisine ve Id'lerine erişiniz.
#########################################################################
# user_movie_df' ten feature'larda olan filmlerden movies_watched listesindeki filmleri movies_watched_df'e atadık.
movies_watched_df = user_movie_df[movies_watched]

#70 filmin sütünlarda olduğu dataframe oluştu.
movies_watched_df.head()
movies_watched_df.shape #(138493, 70) 138.493 kullanıcı bu 70 filmden en az birini izlemiş.

# Her bir kullanıcının bu 70 fimden kaçını izlediğini user_movie_count'a atadık.
user_movie_count = movies_watched_df.T.notnull().sum()

# userId yi indexten çıkarıp değişken yaptık.
user_movie_count = user_movie_count.reset_index()

# Kullanıcıların izledikleri film sayısı sütununa movie_count ismini atadık.
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# random user ile 50 den fazla sayıda aynı filmi izlemiş kullanıcılarının userId ' lerini users_same_movies'e atadık.
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 50]["userId"]

users_same_movies.head()
users_same_movies.count() #1095 kullanıcı
users_same_movies.index #indexlerde userId'ler var

#########################################################################
# Görev-4 Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz.
#########################################################################
# 1. random_user ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız.

# random user ın izlediği filmler ile random user ile 50 den fazla sayıda aynı filmi izlemiş kullanıcıların (1095 kullanıcı)
# izlediği filmleri (userId, movies) olacak şekilde birleştirdik.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

final_df.head()
final_df.shape #(1096, 70) (userId, movies)

final_df.T.corr() #(1096, 1096) korelasyon matrisi oluşturduk.

# Korelasyon matrisini unstack ile dataframe haline getirdik.( "userId" "userId" "korelasyonları" olacak şekilde)
# drop_duplicates() ile dataframe den kopyaların kaldırılmasını sağladık.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

# Aşağıdaki işlemlerle dataframe deki 3 sütuna "userId" "userId" "corr" sütun başlıklarını atadık.
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()

# random user ile korelasyonları 0.60 tan fazla veya eşit olan kullanıcların Id lerini ve korelasyonlarını top_users'a atadık.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.60)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users

# top_users taki kullanıcılar ile rating dosyasında inner join ile birleştirilir.
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
# random user kullanıcısı bu listeden çıkarılır.
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#########################################################################
# Görev 5: Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz.
#########################################################################
# korelasyon x filme verdiği puan formülasyonu ile Weighted Rating hesaplanır.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Filmlere göre grupby yapılır ve Weighted Average Recommendation Score hesaplanır.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

recommendation_df[["movieId"]].nunique() #4767 film önerisi sunulabilir.

# Ağırlıklı ortalaması 3'ten büyük olan filmler score bazlı büyükten küçüğe sıralanır.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3].sort_values("weighted_rating", ascending=False)

# movie dosyasından filmlerin Id si ve ismi çekilir weighted_rating ile birleştirilir. ilk 5 film getirilir.
movies_to_be_recommend.merge(movie[["movieId", "title"]]).head(5)

# movieId  weighted_rating                                              title
# 0     3310         3.384851                                    Kid, The (1921)
# 1     3307         3.384851                                 City Lights (1931)
# 2     1198         3.097463  Raiders of the Lost Ark (Indiana Jones and the...
# 3      318         3.097463                   Shawshank Redemption, The (1994)
# 4      541         3.097463                                Blade Runner (1982)

#########################################################################
# Görev 6: Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin
# adına göre item-based öneri yapınız.
# ▪ 5 öneri user-based
# ▪ 5 öneri item-based
# olacak şekilde 10 öneri yapınız.
#########################################################################

movie = pd.read_csv('pythonProject/datasets/movie.csv')
rating = pd.read_csv('pythonProject/datasets/rating.csv')

# Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınması
movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].\
    sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
# movie_id = 1196

movie_name = movie[movie["movieId"] == movie_id][["title"]].values[0]
# movie_name = ['Star Wars: Episode V - The Empire Strikes Back (1980)']

def item_based_recommender(movie_name, user_movie_df, head=10):
    movie_name = user_movie_df[movie_name]
    # Tüm kullanıcıların oylamalrını görmek için user_movie_df'ten movie_name'i çağırdık.
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False)

movies_from_item_based = item_based_recommender('Star Wars: Episode V - The Empire Strikes Back (1980)', user_movie_df, 20).reset_index()
#  title         0
# 0     Star Wars: Episode V - The Empire Strikes Back...  1.000000
# 1             Star Wars: Episode IV - A New Hope (1977)  0.751519
# 2     Star Wars: Episode VI - Return of the Jedi (1983)  0.707453
# 3     Raiders of the Lost Ark (Indiana Jones and the...  0.466479
# 4     Star Wars: Episode III - Revenge of the Sith (...  0.390386
#                                                  ...       ...

# Kullanıcının daha önce izlemediği filmlerin önerilmesi
recommended_item_based_df = movies_from_item_based.loc[~movies_from_item_based["title"].isin(movies_watched)][:5]
recommended_item_based_df
#                                                title         0
# 2  Star Wars: Episode VI - Return of the Jedi (1983)  0.707453
# 4  Star Wars: Episode III - Revenge of the Sith (...  0.390386
# 5          Indiana Jones and the Last Crusade (1989)  0.387821
# 6      Lord of the Rings: The Two Towers, The (2002)  0.369139
# 7             Star Trek II: The Wrath of Khan (1982)  0.363111

