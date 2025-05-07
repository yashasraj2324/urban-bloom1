// server.js
const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");

const app = express();
const port = 3000;

// Infobip API credentials
const INFOBIP_API_URL = "https://api.infobip.com/sms/2/text/advanced";
const INFOBIP_API_KEY = "a3450df078285d4b0c5e023cb1cfca7b-fe452fd5-6001-4811-9354-a3db7bf9ffee"; // Replace with your Infobip API key

// Set up body parser
app.use(bodyParser.json());

// Route to send SMS
app.post("/send-sms", async (req, res) => {
  const { phoneNumber, plantName, reminderTime } = req.body;

  const message = `Reminder: Water the ${plantName} at ${reminderTime}`;

  // Prepare the data to send to Infobip API
  const payload = {
    messages: [
      {
        from: "PlantCare", // Customize sender name
        to: phoneNumber, // User's phone number
        text: message, // The SMS content
      },
    ],
  };

  try {
    // Send SMS via Infobip API
    const response = await axios.post(INFOBIP_API_URL, payload, {
      headers: {
        "Authorization": `App ${INFOBIP_API_KEY}`,
        "Content-Type": "application/json",
      },
    });

    // Check if the message was sent successfully
    if (response.status === 200) {
      res.status(200).json({ success: true, message: "SMS sent successfully!" });
    } else {
      res.status(500).json({ success: false, message: "Error sending SMS" });
    }
  } catch (error) {
    console.error("Error sending SMS:", error);
    res.status(500).json({ success: false, message: "Error sending SMS" });
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
