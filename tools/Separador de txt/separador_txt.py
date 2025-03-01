import os

def separar_txt(arquivo_entrada): #Separa as linhas de um arquivo .txt em vários arquivos.
    try:
        with open(arquivo_entrada, 'r', encoding='utf-8') as f_entrada:  # Usar utf-8 para suportar acentos e outros caracteres
            linhas = f_entrada.readlines()
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo_entrada}' não encontrado.")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    pasta_saida = os.path.dirname(f"output\{arquivo_entrada}") # Usar a mesma pasta do arquivo de entrada
    nome_base = os.path.splitext(os.path.basename(arquivo_entrada))[0] #Pegar o nome base do arquivo sem a extensão .txt

    for i, linha in enumerate(linhas):
        nome_arquivo_saida = os.path.join(pasta_saida, f"arquivo_{i+1}.txt") # usar o nome base para criar os arquivos
        try:
            with open(nome_arquivo_saida, 'w', encoding='utf-8') as f_saida: # Usar utf-8 para suportar acentos e outros caracteres
                f_saida.write(linha)
        except Exception as e:
            print(f"Erro ao escrever no arquivo '{nome_arquivo_saida}': {e}")

    print(f"Arquivo '{arquivo_entrada}' separado em {len(linhas)} arquivos na pasta '{pasta_saida}'.")

# Exemplo de uso:
if __name__ == "__main__":
    nome_arquivo = "input.txt"
    separar_txt(nome_arquivo)
    print("Processo concluído.")