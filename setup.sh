mkdir -p ~/.streamlit/
echo "[general]
email = \"kanglin.chen@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

# Add the download command
wget -P model https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth