import sys
import time

import graphviz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, ward
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import silhouette_score

class Iris:
    def __init__(self):
        iris_dataset = load_iris()

        # 特徴量
        self.data = iris_dataset.data
        # 特徴量の名前
        self.feature_names = iris_dataset.feature_names
        # クラス
        self.target = iris_dataset.target
        # クラスの名前
        self.target_names = iris_dataset.target_names

        # 特徴量と特徴量の名前をまとめたDataFlame
        # iris_dataset.dataは、iris_dataset["data"], iris_dataset.targetは、iris_dataset["target"]とそれぞれ書くこともできる。
        df_raw = pd.DataFrame(self.data, columns=self.feature_names)

        # "Label"があるdfを作成
        self.df_raw_with_label = df_raw.copy()
        self.df_raw_with_label["Label"] = self.target

    ## 課題1
    def get(self):
        # "Label"があるdfを表示
        return self.df_raw_with_label

    def get_correlation(self):
        # self.df_raw_with_labelからlabelを除いた部分を取り出したものをdf_rawと定義
        # .corr()で相関係数を計算
        df_raw = self.df_raw_with_label.iloc[:, :4]
        return df_raw.corr()

    def pair_plot(self, diag_kind = "hist"):
        # diag_kindが"hist"の時以外はdiag_kwsを指定しない
        if diag_kind == "hist":
            diag_kws={'bins': 20}
        else:
            diag_kws = None

        # "LabelName"列を追加した
        df_raw_with_label_name = self.df_raw_with_label.copy()
        df_raw_with_label_name["LabelName"] = self.target
        for i, target_name in enumerate(self.target_names):
            df_raw_with_label_name["LabelName"][df_raw_with_label_name["LabelName"] == i] = str(target_name)
        # "Label" 列を削除
        df_raw_with_label_name = df_raw_with_label_name.drop("Label", axis=1)


        grr = sns.pairplot(df_raw_with_label_name,
            hue = "LabelName",
            markers="o",
            diag_kws=diag_kws,
            plot_kws={'alpha': 0.8},
            diag_kind = diag_kind
            )

    ## 課題2
    def all_supervised(self, n_neighbors=4):
        # self.dataをローカル変数xとして持っておく
        x = self.data

        # 学習させた訓練モデルを格納する辞書。ここではデータを5分割するから一つのモデル名に対して5つの学習モデルが生成され、格納される。
        self.dict_kfold_trained_models = {}
        # test_scoreを格納するdataflame
        self.df_test_score = pd.DataFrame()

        ## モデル
        # Logistic Regression: ロジスティック回帰(回帰アルゴリズムではなく、クラス分類アルゴリズム)
        # LinearSVC: 線形サポートベクタマシン(回帰アルゴリズムではなく、クラス分類アルゴリズム)
        # DecisionTreeClassifier: 決定木分類機
        # KNeighborsClassifier: k近傍法分類器
        # Linear Regression: 線形モデル
        # RandomForestClassifier: ランダムフォレスト分類木
        # GradientBoostingClassifier: 勾配ブースティング分類器
        # MLPClassifier: 多層パーセプトロン分類器
        self.list_models = [
            LogisticRegression(),
            LinearSVC(),
            DecisionTreeClassifier(),
            KNeighborsClassifier(n_neighbors=n_neighbors),
            LinearRegression(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            MLPClassifier()
            ]

        # StratifiedKFold法のスプリット数
        n_splits = 5
        # KFold法と違い、StratifiedKFold法ではクラスの分類に際してクラスの割合に偏りが出ないように分割が行われる。
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for model in self.list_models:
            # modelの名前をstrとして変換し、printする
            model_name = type(model).__name__
            print(f"=== {model_name} ===")

            # enumerateをすると、インデックス番号、要素の順で取得できる。よって、ここではiにインデックス番号、train_indexおよびtest_indexにk_fold法で分けられたデータの要素が入る。
            for i, (train_index, test_index) in enumerate(skf.split(x, self.target)):
                # k_fold法ではインデックス番号のみがリストとして返される。データではない。
                X_train, y_train = x[train_index], self.target[train_index]
                X_test, y_test = x[test_index], self.target[test_index]
                model.fit(X_train, y_train)

                # 学習済みモデルを辞書に格納
                # もしmodel_nameをキーとしてそのモデルが存在していなかったらmodel_nameをキー、fitさせたmodelを値として追加する。
                if model_name not in self.dict_kfold_trained_models:
                    self.dict_kfold_trained_models[model_name] = [model]
                # もしすでにmodel_nameをキーとしてそのモデルが存在していたら、model_nameをキー、fitさせたmodelを値として追加する。
                # キー = model_nameに対して値 = リストとすることで、一つのmodel_nameに対して複数のmodel(n_splits個数分)を格納できる
                else:
                    self.dict_kfold_trained_models[model_name].append(model)

                # test_scoreを計算
                test_score = model.score(X_test, y_test)
                # i番号を行名, model_nameを列名に指定し、atメソッドで分割番号iのときの各model_nameのtest_scoreを格納
                self.df_test_score.at[i, model_name] = test_score
                # train_scoreを計算
                train_score = model.score(X_train, y_train)
                # \tでタブキー分スペースを開けるという意味になる。
                print(f"test score:{test_score:.3f}\t train score:{train_score:.3f}")

    def get_supervised(self):
        return self.df_test_score

    def best_supervised(self):
        return self.df_test_score.describe().loc["mean"].idxmax(), self.df_test_score.describe().loc["mean"].max()

    # AllSupervised()のときにstratifiedkoldで学習させた5つのモデルのうち,i番目のモデルを用いて重要度を計算し、プロット
    # なお、デフォルトではi=5(5つの学習モデルのうちの5番目)
    def plot_feature_importances_all(self, i=4):
        list_selected_models = ["DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]
        # 新しい辞書に、対象のモデル名(キー)とfitさせたモデル(値)を格納
        dict_selected_trained_models = {
            model_name: self.dict_kfold_trained_models[model_name]
            for model_name in list_selected_models if model_name in self.dict_kfold_trained_models}

        _, axes = plt.subplots(len(list_selected_models), 1, figsize=(5, 12))
        # .itemsによってキーと値の両方をfor文で取得する。つまり、model_nameがキー、modelsに値が出力されていく
        for index, (model_name, models) in enumerate(dict_selected_trained_models.items()):
            model = models[i]
            feature_importances = model.feature_importances_
            axes[index].barh(self.feature_names, feature_importances)
            axes[index].set_xlabel("Feature importance:" + model_name)
        plt.show()

    # kfoldで学習させたDecisionTreeClassifierの5つあるうちi番目の学習モデルの決定木
    def visualize_decision_tree(self, i=4):
        trained_decision_tree_model = self.dict_kfold_trained_models["DecisionTreeClassifier"]
        first_trained_decision_tree_model = trained_decision_tree_model[i]
        return plot_tree(
            first_trained_decision_tree_model,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
        )

    ## 課題3
    def plot_scaled_data(self):
        x = self.data
        self.dict_scalers = {
            "Original": None,
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "Normalizer": Normalizer(),
            }
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # 特徴量のコンビネーション
        list_combinations = [(0, 1), (1, 2), (2, 3), (3, 0)]
        # k_Fold法で分割
        for fold, (train_index, test_index) in enumerate(skf.split(x, self.target)):
            # k_fold法ではインデックス番号のみがリストとして返される。データではない！
            X_train, y_train = x[train_index], self.target[train_index]
            X_test, y_test = x[test_index], self.target[test_index]
            # 各分割データごとにスコアを初期化
            list_fold_scores = []

            for k, (scaler_name, scaler) in enumerate(self.dict_scalers.items()):
                if scaler:
                        # X_trainをfitかつtransform(スケール変換)させる
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                        # fitさせるのはX_trainのみ。y_trainには行わない。またX_testをtransform(スケール変換)する
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test

                    # k番目の分割データのスコアを表示
                model = LinearSVC()
                model.fit(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                train_score = model.score(X_train_scaled, y_train)
                list_fold_scores.append(
                    {"scaler": scaler_name, "test_score": test_score, "train_score": train_score}
                )

            for score in list_fold_scores:
                print(f"{score['scaler']}:\t test score: {score['test_score']:.3f} train score: {score['train_score']:.3f}")

            for i, j in list_combinations:
                fig, axes = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True)
                for k, (scaler_name, scaler) in enumerate(self.dict_scalers.items()):
                    X_train_scaled = X_train if scaler is None else scaler.transform(X_train)
                    X_test_scaled = X_test if scaler is None else scaler.transform(X_test)

                    axes[k].scatter(X_train_scaled[:, i], X_train_scaled[:, j], c="blue", label="Training Set", s=60)
                    axes[k].scatter(X_test_scaled[:, i], X_test_scaled[:, j], c="red", label="Test Set", marker="^", s=60)
                    axes[k].set_title(scaler_name)
                    axes[k].set_xlabel(self.feature_names[i])
                    axes[k].set_ylabel(self.feature_names[j])
                # グラフを表示
                plt.show()
            # 区切り線を表示
            print("=========================================================================")

        # リストをpandas DataFlameに変換
        self.df_scale_scores = pd.DataFrame(list_fold_scores)



    ### irisで次元削減を行い、プロットする場合の汎用関数 引数:(スケーリング手法, 次元削減手法, 主成分量, 入力データ, 出力データ)
    def plot_dimensionality_reduction(self, scaler, reduction_method, n_components, data, target):

        if scaler == None:
            X_scaled = data
        else:
            # スケール変換
            X_scaled = scaler.fit_transform(data)

        # 次元削減
        X_transformed = reduction_method.fit_transform(X_scaled)

        # 削減前と削減後の次元を確認
        print("Original shape: {}".format(str(X_scaled.shape)))
        print("Reduced shape: {}".format(str(X_transformed.shape)))

        ## プロット表示
        # plt.figure(figsize=(8, 8))
        fig, ax = plt.subplots(figsize=(8, 8))
        # 引数はそれぞれx値, y値, label
        mglearn.discrete_scatter(X_transformed[:, 0], X_transformed[:, 1], target)
        plt.legend(self.target_names, loc="best")
        # 縦横比を等しくして表示する
        plt.gca().set_aspect("equal")
        # plt.axis("equal")

        # アスペクト比の調整
        x_range = X_transformed[:, 0].max() - X_transformed[:, 0].min()
        y_range = X_transformed[:, 1].max() - X_transformed[:, 1].min()
        ax.set_aspect(x_range / y_range)

        # 主成分: データにおける分散が大きい方向へのベクトル
        # 第一主成分
        plt.xlabel("First principal component")
        # 第二主成分
        plt.ylabel("Second principal component")
        plt.show()

        ## ヒートマップ表示
        plt.matshow(reduction_method.components_, cmap="viridis")
        # 第一主成分, 第二主成分
        plt.yticks([0, 1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=60, ha="left")
        plt.xlabel("Feature")
        plt.ylabel("Principal components")
        plt.show()

        return (
            pd.DataFrame(X_scaled, columns=self.feature_names),
            pd.DataFrame(X_transformed, columns=[str(i) for i in range(n_components)]),
            reduction_method,
        )

    def plot_pca(self, n_components=2):
        x = self.data
        scaler = StandardScaler()
        reduction_method = PCA(n_components=n_components)
        return self.plot_dimensionality_reduction(
            scaler=scaler,
            reduction_method=reduction_method,
            n_components=n_components,
            data=x,
            target=self.target,
        )

    # NMF: 次元削減に用いられる教師なし学習手法
    # 非負の重み付き和に分解。→幾つもの独立した発生源から得られたデータを重ね合わせて作られるようなデータに対して有効
    def plot_nmf(self, n_components=2):
        x = self.data
        reduction_method = NMF(n_components=n_components, random_state=42)
        return self.plot_dimensionality_reduction(
            scaler=None,
            reduction_method=reduction_method,
            n_components=n_components,
            data=x,
            target=self.target)

    # 多様体学習アルゴリズムのひとつ t-SNEアルゴリズム
    # 訓練データの新たな表現を計算
    def plot_tsne(self):
        x = self.data
        scaler = None
        tsne = TSNE(random_state=42)
        X_scaled = tsne.fit_transform(x)
        plt.figure(figsize=(10, 10))
        plt.xlim(X_scaled[:, 0].min(), X_scaled[:, 0].max() + 1)
        plt.ylim(X_scaled[:, 1].min(), X_scaled[:, 1].max() + 1)
        for i in range(len(x)):
            # actually plot the digits as text instead of using scatter
            plt.text(X_scaled[i, 0], X_scaled[i, 1], str(self.target[i]), fontdict={"weight": "bold", "size": 9})
        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")

    # plotを行うための関数
    def plot_method(self, x, y, i_value, x_feature, y_feature, cluster_method=None, activate_cluster_center = False):
        plt.figure()
        for i, marker, color in zip(i_value, ["o", "^", "v"], ["blue", "red", "green"]):
            plt.scatter(x[y == i, x_feature], x[y == i, y_feature], marker=marker, c=color)
            plt.xlabel(f"Feature{int(x_feature)}")
            plt.ylabel(f"Feature{int(y_feature)}")
        if activate_cluster_center == True:
            # クラスタの中心点をプロット
            plt.scatter(cluster_method.cluster_centers_[:, x_feature], cluster_method.cluster_centers_[:, y_feature], marker="*", c="black", s=200)
            plt.show()
        else:
            pass

    # クラスタリング手法の一つ KMeans
    def plot_k_means(self):
        x = self.data
        # 正規化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)
        # 新たに加えた(2023/05/09)
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X_transformed)

        # Kmeans
        predicted_labels = kmeans.labels_
        print(f"KMeans法で予測したラベル:\n {predicted_labels}")
        predicted_plot = self.plot_method(x=X_transformed, y=predicted_labels, cluster_method=kmeans, i_value=range(3), x_feature = 0, y_feature = 1, activate_cluster_center = True)

        # 実際のデータ
        actual_labels = self.target
        print(f"実際のラベル:\n {actual_labels}")
        actual_plot = self.plot_method(x=X_transformed, y=actual_labels, cluster_method=kmeans, i_value=range(3), x_feature = 0, y_feature = 1, activate_cluster_center = True)

    # 凝集型クラスタリング: 個々のデータポイントをそれぞれ個別のクラスタとして開始し、最も類似した2つのクラスタを併合していく
    # これによって階層的クラスタリングが実現でき、デンドログラムで可視化できる
    def plot_dendrogram(self, truncate=False):
        df_raw = self.df_raw_with_label.iloc[:, :4]
        iris_war = linkage(df_raw, method="ward", metric="euclidean")

        if truncate == True:
            dendrogram(iris_war, labels=np.arange(len(self.target)), truncate_mode="lastp", p=10)
        else:
            dendrogram(iris_war, labels=np.arange(len(self.target)))
        ax = plt.gca()
        bounds = ax.get_xbound()
        ax.plot(bounds, [10, 10], "--", c="k")
        ax.plot(bounds, [6, 6], "--", c="k")
        ax.text(bounds[1], 10, "3 clusters", va="center", fontdict={"size": 12})
        ax.text(bounds[1], 6, "4 clusters", va="center", fontdict={"size": 12})
        plt.xlabel("Sample index")
        plt.ylabel("Cluster Distance")
        plt.show()

    # 凝集的クラスタリングやKMeans法よりも遅いが、大きなデータセットにも適用可能なDBScan
    def plot_dbscan(self, scaling=False, eps = 0.5, min_samples = 5):
        x = self.data

        if scaling == True:
            # StandardScalerを用いると、平均0, 分散1になるようにスケール変換を行うので、クラスタが削減される。
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(x)
        else:
            X_scaled = x

        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        dbscan.fit(X_scaled)
        clusters = dbscan.fit_predict(X_scaled)
        print(f"Clusters memberships:\n {clusters}")
        return self.plot_method(x=X_scaled, y=clusters, cluster_method=dbscan, i_value=[-1, 0, 1], x_feature = 2, y_feature = 3, activate_cluster_center=False)