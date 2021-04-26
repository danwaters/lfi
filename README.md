# Lyrics from Image 
See the [paper](https://github.com/danwaters/lfi/blob/main/Dan%20Waters%20-%20CSCE%205218%20Final%20Project%20Report.pdf)

# Deployed Architecture
![Blank diagram](https://user-images.githubusercontent.com/780735/116029988-2b446900-a620-11eb-84b3-db2a855a7289.png)

# Colab Outputs
* [Natural language](https://github.com/danwaters/lfi/blob/main/NL%20Model%20Training%20Outputs.pdf)
* [Image classification](https://github.com/danwaters/lfi/blob/main/IC%20Model%20Training%20Outputs.pdf)

# Test it out yourself
* **URL**: https://us-central1-dogbot-298321.cloudfunctions.net/create_lyrics
* **Method**: POST
* **Headers**: Content-Type: application/json
* **Body**: `(application/json)`
`{"url": "https://<your_image_url>", "seed": "<text to start the song with>"}`

Example response:
```
{
    "image_url": "https://www.rollingstone.com/wp-content/uploads/2018/06/rs-7349-20121003-beatles-1962-624x420-1349291947.jpg?resize=1800,1200&w=1800",
    "predicted_class": "zappa",
    "predicted_lyrics": " I his gonna along around all me\r\n twirly see up   can\r\n doggie speak tu going long like\r\n in a   might speak\r\n his gonna crop get now i\r\n his gonna along around all me\r\n twirly see up   can\r\n doggie speak tu going long like\r\n in a   might speak\r\n his gonna crop get now i\r\n his gonna along around all me\r\n twirly see up   can\r\n doggie speak tu going long like\r\n in a   might speak\r\n his gonna crop get now i\r\n his gonna along around all me\r\n twirly see up "
}
```
