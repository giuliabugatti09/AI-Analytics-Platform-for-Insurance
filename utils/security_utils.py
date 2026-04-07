import requests
from urllib.parse import urlparse
import streamlit as st
import ipaddress
import socket
from pathlib import Path
import re

def obter_paginas_streamlit_validas():
    """
    Escaneia a pasta 'pages' para encontrar todas as páginas válidas da aplicação.
    Streamlit usa o nome do arquivo .py como o caminho da URL.
    """
    # A página principal (home) sempre corresponde ao caminho raiz '/'
    paginas_validas = {"/"} 
    
    pages_dir = Path("pages")
    if pages_dir.is_dir():
        # Itera sobre todos os arquivos .py na pasta 'pages'
        for page_file in pages_dir.glob("*.py"):
            # Streamlit permite prefixos numéricos (ex: "1_Heatmap.py") para ordenação.
            # Vamos remover esses prefixos para obter o nome real da página.
            page_name = re.sub(r'^\d+[_.\s-]*', '', page_file.stem)
            paginas_validas.add(f"/{page_name}")
            
    return paginas_validas

def obter_hosts_locais_permitidos():
    """
    Gera dinamicamente um conjunto de hosts locais permitidos.
    Isso inclui localhost, o hostname da máquina e seu IP local.
    """
    hosts = {"localhost", "127.0.0.1"}
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        hosts.add(hostname)
        hosts.add(ip_address)
    except socket.gaierror:
        # Pode falhar em algumas configurações de rede, mas localhost ainda funcionará
        st.warning("Não foi possível resolver o hostname local dinamicamente.")
    return hosts

def eh_ip_privado(hostname):
    """
    Verifica se o hostname resolve para um IP privado, para evitar escaneamento da rede interna.
    """
    try:
        # Tenta resolver o hostname para um IP
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private
    except (socket.gaierror, ValueError):
        # Se não for um IP válido ou não puder ser resolvido, consideramos inseguro por padrão
        return True

def fazer_requisicao_segura(url: str, hosts_permitidos: set, protocolos_permitidos: set, paginas_validas: set):
    """
    Faz uma requisição HTTP/HTTPS de forma segura, validando protocolo, host E CAMINHO.
    """
    try:
        parsed_url = urlparse(url)
        
        # 1. Validação de Protocolo
        if parsed_url.scheme not in protocolos_permitidos:
            return None, f"❌ Erro: Protocolo '{parsed_url.scheme}' não é permitido."

        # 2. Validação de Host
        hostname = parsed_url.hostname
        if not hostname or hostname not in hosts_permitidos:
            if eh_ip_privado(hostname):
                 return None, f"❌ Erro: O host '{hostname}' não é permitido ou aponta para um endereço de rede interna."
            return None, f"❌ Erro: Host '{hostname}' não está na lista de hosts permitidos."

        # --- NOVA VALIDAÇÃO DE CAMINHO ---
        # Normaliza o caminho: se for vazio (ex: 'http://localhost:8501'), trata como '/'
        caminho = parsed_url.path if parsed_url.path else "/"
        if caminho not in paginas_validas:
            return None, f"❌ Erro: A página '{caminho}' não existe na aplicação."
        # --- FIM DA NOVA VALIDAÇÃO ---

        # 3. Realizar a Requisição
        response = requests.get(url, timeout=5)
        
        # Mesmo que a requisição funcione, se o Streamlit retornar 'Page not found', o status pode ser 200.
        # Vamos checar o conteúdo da resposta.
        if "Page not found" in response.text and "The page that you have requested does not seem to exist" in response.text:
             return None, f"❌ Erro: A URL retornou uma página de 'Não Encontrado' do Streamlit."
        
        response.raise_for_status()
        
        return response.text, f"✅ Sucesso: Requisição para '{url}' realizada."

    except requests.exceptions.RequestException as e:
        return None, f"❌ Erro de rede ao acessar a URL: {e}"
    except Exception as e:
        return None, f"❌ Ocorreu um erro inesperado: {e}"
    
def normalizar_url(url_string: str) -> str:
    """Garante que a URL tenha um protocolo, adicionando 'https://' como padrão se necessário."""
    url_string = url_string.strip()
    if not url_string:
        return ""
    # Adiciona http:// para localhost para garantir que a requisição funcione localmente
    if url_string.startswith("localhost"):
        return f"http://{url_string}"
    parsed_url = urlparse(url_string)
    if not parsed_url.scheme:
        return f"https://{url_string}"
    return url_string
