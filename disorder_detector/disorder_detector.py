from transformers import RobertaTokenizer, XLMRobertaTokenizer, XLMRobertaModel

from text_classification import load_data, split_data, to_tensor, main, TrainingConfig, plot_report

# Define the label dictionary
label_dict = {
    'Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down': 0,
    'Feeling-down-depressed-or-hopeless': 1,
    'Feeling-tired-or-having-little-energy': 2,
    'Little-interest-or-pleasure-in-doing ': 3,
    'Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual': 4,
    'Poor-appetite-or-overeating': 5,
    'Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way': 6,
    'Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television': 7,
    'Trouble-falling-or-staying-asleep-or-sleeping-too-much': 8
}

training_configs = TrainingConfig(
    max_length=128,
    batch_size=16,
    test_size=0.2,
    num_epochs=4,
    learning_rate=2e-5,
)

data_df = load_data("primate_data.csv")
X_train, X_test, y_train, y_test = split_data(data_df, label_dict=label_dict)
tensored_data = to_tensor(X_train, X_test, y_train, y_test)

def train_XLM_Roberta():
    data_df = load_data("primate_data.csv")
    X_train, X_test, y_train, y_test = split_data(data_df, label_dict=label_dict)
    tensored_data = to_tensor(X_train, X_test, y_train, y_test)

    xlm_roberta_large_languge_model = 'xlm-roberta-large-finetuned-conll03-english'

    xlm_roberta_large_reports, xlm_roberta_large_model, xlm_roberta_large_tokenizer, xlm_roberta_large_device = main(tensored_data,
                                                                                                                    xlm_roberta_large_languge_model,
                                                                                                                    training_configs=training_configs,
                                                                                                                    layer='23',
                                                                                                                    tokenizer_class=XLMRobertaTokenizer,
                                                                                                                    bert_class=XLMRobertaModel)

    plot_report(xlm_roberta_large_reports)


def train_BERT():
    data_df = load_data("primate_data.csv")
    X_train, X_test, y_train, y_test = split_data(data_df, label_dict=label_dict)
    tensored_data = to_tensor(X_train, X_test, y_train, y_test)

    bertـlanguge_model = 'bert-base-uncased'

    bertـreports, bertـmodel, bertـtokenizer, bertـdevice = main(
        tensored_data, 
        bertـlanguge_model, 
        training_configs=training_configs,
        layer='11')

    plot_report(bertـreports)


def distil_BERT():
    data_df = load_data("primate_data.csv")
    X_train, X_test, y_train, y_test = split_data(data_df)
    tensored_data = to_tensor(X_train, X_test, y_train, y_test)
    distil_bert__language_model = 'distilbert-base-uncased'

    distilbert_reports = main(
        tensored_data,
        distil_bert__language_model,
        training_configs=training_configs,
        layer='9')
    plot_report(distilbert_reports)

if __name__ == '__main__':
    print("----------------- Train XLM Roberta model------------------")
    train_XLM_Roberta()
    print("----------------- Train BERT model ----------------------")
    train_BERT()
    print("----------------- Train distil BERT model ----------")
    distil_BERT()