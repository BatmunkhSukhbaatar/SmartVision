import requests

def synthesize(text):
    url = "https://api.chimege.com/v1.2/synthesize"
    headers = {
        'Content-Type': 'plain/text',
        'Token': '245e07418df31091e0411264401eade33bd7cb558c9ea38ece395caf022daf80',
    }

    r = requests.post(
        url, data=text.encode('utf-8'), headers=headers)

    with open("PPT.wav", 'wb') as out:
        out.write(r.content)

def normalize(text):
    url = "https://api.chimege.com/v1.2/normalize-text"
    headers = {
        'Content-Type': 'plain/text',
        'Token': '245e07418df31091e0411264401eade33bd7cb558c9ea38ece395caf022daf80',
    }

    r = requests.post(
        url, data=text.encode('utf-8'), headers=headers)

    return r.content.decode("utf-8")

# text = 'УБ-т 2021-04-13-нд 33000 тээврийн хэрэгсэл замын хөдөлгөөнд оролцжээ.'
text = 'Хүн 4 метрийн зайтай ойртож байна. Та болгоомжтой байна уу ?'

print(synthesize(normalize(text)))

