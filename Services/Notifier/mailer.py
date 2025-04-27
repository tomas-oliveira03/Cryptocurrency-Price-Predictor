import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime

def sendCryptoEmail(SHOW_LOGS, senderEmail, senderPassword, smtpServer, smtpPort, coinName, currentPrice, percentageChange, priceDifferential, recipientEmail):
    subject = f"Cryptocurrency Update: {coinName} at ${currentPrice:,.2f}"
    body = createCryptoEmailTemplate(coinName, currentPrice, percentageChange, priceDifferential)

    message = MIMEMultipart()
    message["From"] = senderEmail
    message["To"] = recipientEmail
    message["Subject"] = subject
    
    message.attach(MIMEText(body, "html"))
    
    try:
        # Create SMTP session
        server = smtplib.SMTP(smtpServer, smtpPort)
        server.starttls()
        
        # Login to sender email
        server.login(senderEmail, senderPassword)
        
        # Send email
        server.sendmail(senderEmail, recipientEmail, message.as_string())
        
        server.quit()
        if SHOW_LOGS: print(f"✅ Email successfully sent to {recipientEmail}")
        
    except Exception as e:
        if SHOW_LOGS: print(f"❌ Error sending email to {recipientEmail}: {e}")


def createCryptoEmailTemplate(coinName, currentPrice, percentageChange, priceDifferential):
    color = "green" if percentageChange >= 0 else "red"
    sign = "+" if percentageChange >= 0 else ""
    
    htmlTemplate = f"""
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
                    <strong>{coinName}</strong>
                </div>
                <div class="price">
                    ${currentPrice:,.2f}
                </div>
                <div class="change">
                    {sign}{percentageChange:.2f}% (${priceDifferential:,.2f})
                </div>
            </div>
            <div class="footer">
                This is an automated cryptocurrency update.<br>Data as of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    return htmlTemplate
