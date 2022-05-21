const tf = require('@tensorflow/tfjs-node');
var bodyParser = require('body-parser')
const express = require('express')
const app = express()
const port = 3000
// parse application/x-www-form-urlencoded
app.use(bodyParser.urlencoded({ extended: false }))

// parse application/json
app.use(bodyParser.json())

app.post('/prediksi', (req, res) => {
    console.log(req.body);
    // d = tf.tensor(d, [1, 259, 1])
    // d.print()
    // let z = model.predict(d)
    // z.print()
    // let index = z.argMax(1).dataSync()
    // console.log(index);
    // alert(`Emosi Anda: ${label[index]}`)
  res.send('Hello World!')
})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})