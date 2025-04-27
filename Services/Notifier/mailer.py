import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import datetime
import sys
import pathlib

# Add project root to Python path to be able to import utils
PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def sendCryptoEmail(SHOW_LOGS, senderEmail, senderPassword, smtpServer, smtpPort, coinName, currentPrice, percentageChange, priceDifferential, recipientEmail, alertCondition=None, targetPrice=None, monitoredPriceType=None):
    # Create subject line with alert type and price type
    priceType = "Predicted" if monitoredPriceType == "PREDICTED" else "Real-Time"
    alertType = "above" if alertCondition == "ABOVE" else "below"
    subject = f"🚨 {priceType} Alert: {coinName} ${currentPrice:,.2f} {alertType.upper()} target"
    
    # Create email message
    message = MIMEMultipart()
    message["From"] = senderEmail
    message["To"] = recipientEmail
    message["Subject"] = subject
    
    # Attach logo if available
    logo_path = os.path.join(PROJECT_ROOT, "Services", "utils", "crypto-logos", f"{coinName}.png")
    logo_cid = None
    
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as img_file:
            img_data = img_file.read()
            image = MIMEImage(img_data)
            logo_cid = f'{coinName}_logo'
            image.add_header('Content-ID', f'<{logo_cid}>')
            image.add_header('Content-Disposition', 'inline', filename=f"{coinName}.png")
            message.attach(image)
            if SHOW_LOGS: print(f"✅ Logo attached for {coinName}")
    else:
        if SHOW_LOGS: print(f"❌ No logo found for {coinName} at {logo_path}")
    
    # Create HTML body with logo reference if logo was found
    body = createCryptoEmailTemplate(coinName, currentPrice, alertCondition, targetPrice, monitoredPriceType, logo_cid)
    
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


def createCryptoEmailTemplate(coinName, currentPrice, alertCondition, targetPrice, monitoredPriceType, logo_cid=None):
    # Define alert details
    condition_text = "above" if alertCondition == "ABOVE" else "below"
    price_type = "Predicted" if monitoredPriceType == "PREDICTED" else "Real-Time"
    
    # Alert message
    alert_message = f"Your {price_type} price alert for {coinName} has been triggered."
    
    # Logo HTML - only include if we have a logo
    logo_html = ""
    if logo_cid:
        logo_html = f"""
        <div class="logo-container">
            <img src="cid:{logo_cid}" alt="{coinName} logo" class="coin-logo" />
        </div>
        """
    
    htmlTemplate = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Roboto, Arial, sans-serif;
                background-color: #f4f4f4;
                color: #333;
                padding: 20px;
                margin: 0;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background-color: #ffffff;
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 30px;
                color: #0078d4;
                letter-spacing: 0.5px;
                border-bottom: 2px solid #eaeaea;
                padding-bottom: 15px;
            }}
            .alert-message {{
                background-color: #f0f7ff;
                border-left: 4px solid #0078d4;
                padding: 15px;
                margin-bottom: 25px;
                border-radius: 8px;
                font-size: 16px;
                line-height: 1.6;
                color: #333;
            }}
            .price-comparison {{
                display: flex;
                justify-content: space-around;
                margin: 30px 0;
                text-align: center;
            }}
            .price-box {{
                padding: 20px;
                border-radius: 12px;
                background-color: #f8f8f8;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                width: 45%;
                border: 1px solid #eaeaea;
            }}
            .price-label {{
                font-size: 16px;
                color: #666;
                margin-bottom: 10px;
                font-weight: 500;
            }}
            .price-value {{
                font-size: 32px;
                font-weight: 800;
                letter-spacing: 0.5px;
                color: #0078d4;
            }}
            .coin-name {{
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                margin: 20px 0 10px;
                color: #0078d4;
            }}
            .alert-type {{
                text-align: center;
                font-size: 18px;
                margin: 25px 0;
                padding: 10px;
                background-color: #f0f7ff;
                border-radius: 8px;
                font-weight: 500;
                color: #333;
            }}
            .footer {{
                text-align: center;
                font-size: 13px;
                margin-top: 30px;
                color: #666;
                padding-top: 20px;
                border-top: 1px solid #eaeaea;
            }}
            .time-stamp {{
                font-style: italic;
                margin-top: 10px;
                color: #999;
            }}
            .price-type {{
                display: inline-block;
                background-color: #0078d4;
                color: #fff;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 14px;
                margin-bottom: 15px;
                font-weight: 600;
            }}
            .logo-container {{
                text-align: center;
                margin: 0 auto 10px;
            }}
            .coin-logo {{
                max-width: 80px;
                max-height: 80px;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                {price_type} Cryptocurrency Alert 🚨
            </div>
            
            <div class="alert-message">{alert_message}</div>
            
            {logo_html}
            <div class="coin-name">{coinName}</div>
            
            <div class="alert-type">
                Price is now {condition_text} your target of ${targetPrice:,.2f}
            </div>
            
            <div class="price-comparison">
                <div class="price-box">
                    <div class="price-label">Current Price</div>
                    <div class="price-value">${currentPrice:,.2f}</div>
                </div>
                
                <div class="price-box">
                    <div class="price-label">Your Target</div>
                    <div class="price-value">${targetPrice:,.2f}</div>
                </div>
            </div>
            
            <div class="footer">
                This is an automated cryptocurrency alert notification.
                <div class="time-stamp">
                    Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return htmlTemplate
