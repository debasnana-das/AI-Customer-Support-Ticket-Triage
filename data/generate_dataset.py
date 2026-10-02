import random
from pathlib import Path
import pandas as pd

random.seed(42)
OUT = Path(__file__).resolve().parent / 'tickets.csv'

patterns = {
    'billing': [
        'I was charged twice for my subscription',
        'my invoice has an incorrect amount',
        'please explain this payment on my card',
        'I need a refund for the last billing cycle',
        'the monthly charge is higher than expected',
        'my discount was not applied to the invoice',
    ],
    'technical': [
        'the application crashes when I upload a file',
        'the website shows an error after login',
        'the API returns a server error intermittently',
        'the mobile app freezes on the checkout page',
        'I cannot connect to the service from my laptop',
        'the dashboard is loading forever',
    ],
    'account': [
        'I cannot reset my password',
        'my account is locked after several login attempts',
        'please change the email address on my account',
        'I need help verifying my identity',
        'I forgot which phone number is linked to my account',
        'my profile details are incorrect',
    ],
    'product': [
        'how do I use the new reporting feature',
        'the export option is missing from my plan',
        'I want to know whether this feature is supported',
        'please explain how the workflow automation works',
        'the search feature does not return expected results',
        'I need guidance on configuring the product',
    ],
}
urgent_phrases = {
    'high': ['urgent', 'immediately', 'critical', 'blocked', 'down', 'cannot work', 'production impact'],
    'medium': ['today', 'important', 'soon', 'deadline', 'affecting my work'],
    'low': ['when possible', 'just a question', 'no rush', 'for information', 'whenever convenient'],
}
rows = []
for i in range(800):
    category = random.choice(list(patterns))
    text = random.choice(patterns[category])
    urgency = random.choices(['low','medium','high'], weights=[0.35,0.40,0.25])[0]
    phrase = random.choice(urgent_phrases[urgency])
    if phrase in text:
        phrase = random.choice(urgent_phrases[urgency])
    text = f"{text}. This is {phrase} for me."
    resolution = random.choice([
        'pending review', 'article sent', 'specialist assigned', 'resolved', 'refund issued'
    ])
    rows.append({
        'ticket_id': f'TKT-{i+1:04d}',
        'text': text,
        'category': category,
        'urgency': urgency,
        'resolution': resolution,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f'Wrote {len(df)} rows to {OUT}')
