import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# Carregar os dados
train_df = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')
test_df = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
sample_submission_df = pd.read_csv('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv')

# Pré-processamento
def preprocessar_texto(texto):
    return texto.lower()

train_df['prompt_processado'] = train_df['prompt'].apply(preprocessar_texto)
train_df['response_a_processado'] = train_df['response_a'].apply(preprocessar_texto)
train_df['response_b_processado'] = train_df['response_b'].apply(preprocessar_texto)

# Características adicionais
train_df['response_a_length'] = train_df['response_a'].apply(len)
train_df['response_b_length'] = train_df['response_b'].apply(len)

# Encoder para a coluna 'language'
le = LabelEncoder()
train_df['language_encoded'] = le.fit_transform(train_df['language'])

test_df['prompt_processado'] = test_df['prompt'].apply(preprocessar_texto)
test_df['response_a_processado'] = test_df['response_a'].apply(preprocessar_texto)
test_df['response_b_processado'] = test_df['response_b'].apply(preprocessar_texto)
test_df['response_a_length'] = test_df['response_a'].apply(len)
test_df['response_b_length'] = test_df['response_b'].apply(len)

# Vetorização TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # Limitar o número de características para economizar memória
tfidf_train_prompt = tfidf_vectorizer.fit_transform(train_df['prompt_processado'])
tfidf_train_response_a = tfidf_vectorizer.transform(train_df['response_a_processado'])
tfidf_train_response_b = tfidf_vectorizer.transform(train_df['response_b_processado'])
tfidf_test_prompt = tfidf_vectorizer.transform(test_df['prompt_processado'])
tfidf_test_response_a = tfidf_vectorizer.transform(test_df['response_a_processado'])
tfidf_test_response_b = tfidf_vectorizer.transform(test_df['response_b_processado'])

# Calcular similaridade de coseno
cosine_sim_a = []
cosine_sim_b = []

for i in range(len(train_df)):
    prompt_vec = tfidf_train_prompt[i]
    response_a_vec = tfidf_train_response_a[i]
    response_b_vec = tfidf_train_response_b[i]

    cosine_sim_a.append(cosine_similarity(prompt_vec, response_a_vec)[0][0])
    cosine_sim_b.append(cosine_similarity(prompt_vec, response_b_vec)[0][0])

train_df['cosine_sim_a'] = cosine_sim_a
train_df['cosine_sim_b'] = cosine_sim_b

# Calcular similaridade de coseno para o conjunto de teste
cosine_sim_test_a = []
cosine_sim_test_b = []

for i in range(len(test_df)):
    prompt_vec = tfidf_test_prompt[i]
    response_a_vec = tfidf_test_response_a[i]
    response_b_vec = tfidf_test_response_b[i]

    cosine_sim_test_a.append(cosine_similarity(prompt_vec, response_a_vec)[0][0])
    cosine_sim_test_b.append(cosine_similarity(prompt_vec, response_b_vec)[0][0])

test_df['cosine_sim_a'] = cosine_sim_test_a
test_df['cosine_sim_b'] = cosine_sim_test_b

# Número de Palavras Únicas
train_df['prompt_unique_words'] = train_df['prompt'].apply(lambda x: len(set(x.split())))
train_df['response_a_unique_words'] = train_df['response_a'].apply(lambda x: len(set(x.split())))
train_df['response_b_unique_words'] = train_df['response_b'].apply(lambda x: len(set(x.split())))

test_df['prompt_unique_words'] = test_df['prompt'].apply(lambda x: len(set(x.split())))
test_df['response_a_unique_words'] = test_df['response_a'].apply(lambda x: len(set(x.split())))
test_df['response_b_unique_words'] = test_df['response_b'].apply(lambda x: len(set(x.split())))

# Combinar as características numéricas e TF-IDF usando hstack
X_train = hstack([tfidf_train_prompt, tfidf_train_response_a, tfidf_train_response_b, csr_matrix(train_df[[
    'response_a_length', 'response_b_length', 'language_encoded', 'cosine_sim_a', 'cosine_sim_b',
    'prompt_unique_words', 'response_a_unique_words', 'response_b_unique_words'
]].values)])

X_test = hstack([tfidf_test_prompt, tfidf_test_response_a, tfidf_test_response_b, csr_matrix(test_df[[
    'response_a_length', 'response_b_length', 'cosine_sim_a', 'cosine_sim_b',
    'prompt_unique_words', 'response_a_unique_words', 'response_b_unique_words'
]].values)])

y_train = train_df['winner'].map({'model_a': 0, 'model_b': 1})  # Mapear a coluna 'winner' para valores numéricos

# Dividir os dados em treino e teste
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definir os modelos LightGBM
modelo_1 = lgb.LGBMClassifier(n_estimators=20, device='gpu', random_state=42)
modelo_2 = lgb.LGBMClassifier(n_estimators=25, learning_rate=0.01, device='gpu', random_state=42)
modelo_3 = lgb.LGBMClassifier(n_estimators=40, learning_rate=0.005, device='gpu', random_state=42)
modelo_4 = lgb.LGBMClassifier(n_estimators=45, learning_rate=0.001, device='gpu', random_state=42)  # Quarto modelo

# Criar o ensemble utilizando VotingClassifier
ensemble = VotingClassifier(estimators=[('lgb1', modelo_1), ('lgb2', modelo_2), ('lgb3', modelo_3), ('lgb4', modelo_4)], voting='soft')
ensemble.fit(X_train_split, y_train_split)

# Fazer previsões com o ensemble
y_pred = ensemble.predict(X_test_split)

# Avaliar o modelo
print("Acurácia do ensemble:", accuracy_score(y_test_split, y_pred))
print("Relatório de Classificação do ensemble:\n", classification_report(y_test_split, y_pred))

# Ajustar o conjunto de teste para corresponder ao número de características do modelo treinado
num_missing_features = X_train.shape[1] - X_test.shape[1]
if num_missing_features > 0:
    missing_features = hstack([X_test, csr_matrix((X_test.shape[0], num_missing_features))])
else:
    missing_features = X_test

# Prever no conjunto de teste
test_df['winner'] = ensemble.predict(missing_features)

# Gerar a submissão
submission_df = sample_submission_df.copy()
submission_df['winner'] = test_df['winner'].map({0: 'model_a', 1: 'model_b'})  # Mapear de volta para os valores originais
submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"Arquivo de submissão salvo em: {submission_path}")
