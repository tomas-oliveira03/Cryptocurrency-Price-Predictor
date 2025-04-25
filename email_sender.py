import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime

def send_email(subject, body, sender_email, sender_password, recipient_email="tomasoliveira.dev@gmail.com", smtp_server="smtp.gmail.com", smtp_port=587, is_html=False):
    """
    Send an email using SMTP
    
    Args:
        subject (str): Email subject
        body (str): Email body content
        sender_email (str): Email address of the sender
        sender_password (str): Password for the sender's email
        recipient_email (str): Email address of the recipient
        smtp_server (str): SMTP server address
        smtp_port (int): SMTP server port
        is_html (bool): Whether the body is HTML content
    """
    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    
    # Add body to email
    content_type = "html" if is_html else "plain"
    message.attach(MIMEText(body, content_type))
    
    try:
        # Create SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        
        # Login to sender email
        server.login(sender_email, sender_password)
        
        # Send email
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)
        
        # Close session
        server.quit()
        print(f"Email successfully sent to {recipient_email}")
        
    except Exception as e:
        print(f"Error sending email: {e}")

def create_crypto_email_template(coin_name, current_price, percentage_change, price_differential):
    """
    Creates an HTML email template for cryptocurrency updates
    
    Args:
        coin_name (str): Name of the cryptocurrency (e.g., "BTC")
        current_price (float): Current price of the cryptocurrency
        percentage_change (float): Percentage change (positive or negative)
        price_differential (float): Price change amount
        
    Returns:
        str: HTML template string
    """
    # Determine color based on percentage change
    color = "green" if percentage_change >= 0 else "red"
    sign = "+" if percentage_change >= 0 else ""
    
    html_template = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #1a1f2e;
                color: white;
                padding: 20px;
                margin: 0;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background-color: #242a3a;
                border-radius: 10px;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                font-size: 24px;
                margin-bottom: 20px;
                color: #3498db;
            }}
            .coin-info {{
                text-align: center;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 8px;
                background-color: #2c3347;
            }}
            .price {{
                font-size: 28px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .change {{
                font-size: 16px;
                color: {color};
                margin-bottom: 10px;
            }}
            .footer {{
                text-align: center;
                font-size: 12px;
                margin-top: 20px;
                color: #888;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                Cryptocurrency Update
            </div>
            <div class="coin-info">
                <div>
                    <strong>{coin_name}</strong>
                </div>
                <div class="price">
                    ${current_price:,.2f}
                </div>
                <div class="change">
                    {sign}{percentage_change:.2f}% (${price_differential:,.2f})
                </div>
            </div>
            <div class="footer">
                This is an automated cryptocurrency update. Data as of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    return html_template

if __name__ == "__main__":
    print("Cryptocurrency Email Notification")
    print("--------------------------------")
    sender_email = os.getenv("GMAIL_USERNAME")
    sender_password = os.getenv("GMAIL_PASSWORD_SMTP")
    
    # Sample cryptocurrency data
    coin_name = "BTC"
    current_price = 91944.00
    percentage_change = 2.78  # positive value means price went up
    price_differential = 2481.00  # positive value means price went up
    
    # Create email subject and body
    subject = f"Cryptocurrency Update: {coin_name} at ${current_price:,.2f}"
    body = create_crypto_email_template(coin_name, current_price, percentage_change, price_differential)
    
    recipient = "tomasoliveira.dev@gmail.com"
    print(f"Sending cryptocurrency update email to {recipient}...")
    send_email(subject, body, sender_email, sender_password, recipient, is_html=True)
