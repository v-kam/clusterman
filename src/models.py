from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-small"  # "text-embedding-3-large"  # 
GPT4O_DEPLOYMENT_NAME = "gpt-4o"
GPT35_DEPLOYMENT_NAME = "gpt-35-turbo-16k"


def gpt4o(temperature=0.3, **kwargs) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=GPT4O_DEPLOYMENT_NAME,
        temperature=temperature,
        **kwargs
    )


def gpt35(temperature=0.1, **kwargs) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=GPT35_DEPLOYMENT_NAME,
        temperature=temperature,
        **kwargs
    )


def emb_small(**kwargs) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
        **kwargs
    )
