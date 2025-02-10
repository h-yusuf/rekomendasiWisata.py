# Project Setup Guide

## 1. Setting Up the Virtual Environment

### Creating a Virtual Environment
Run the following command to create a virtual environment:
```bash
python -m venv venv
```

### Activating the Virtual Environment
For Linux/Mac:
```bash
source venv/bin/activate
```
For Windows (PowerShell):
```powershell
venv\Scripts\Activate
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Saving Installed Packages
```bash
pip freeze > requirements.txt
```

---
## 2. Connecting to the Server via SSH

### Using SSH Key Authentication
To connect to your server, use:
```bash
ssh -i your-key.pem ec2-user@your-server-ip
```
Replace `your-key.pem` with your private key file and `your-server-ip` with your EC2 instance IP address.

### Setting Proper Permissions for SSH Key
If you encounter a permission error, run:
```bash
chmod 400 your-key.pem
```

---
## 3. Checking Server Status

### Checking Nginx Status
```bash
sudo systemctl status nginx
```

### Restarting Nginx
```bash
sudo systemctl restart nginx
```

### Checking Flask Application Status (if running as a systemd service)
```bash
sudo systemctl status flask-app
```

### Restarting Flask Application
```bash
sudo systemctl restart flask-app
```

### Checking Running Processes
```bash
ps aux | grep flask
```

---
## 4. Additional Server Commands

### Checking Open Ports
```bash
netstat -tulnp | grep LISTEN
```

### Viewing Server Logs
For Nginx:
```bash
sudo journalctl -u nginx --no-pager | tail -20
```
For Flask (if using systemd):
```bash
sudo journalctl -u flask-app --no-pager | tail -20
```

