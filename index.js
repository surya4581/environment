// script.js
document.addEventListener('DOMContentLoaded', function () {
    const votingForm = document.getElementById('voting-form');
    const voteButton = document.getElementById('vote-btn');
    const greetingMessage = document.getElementById('greeting-message');

    votingForm.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent form submission
        const selectedParty = votingForm.elements['party'].value;

        // Simulate voting process
        // You can replace this with actual backend logic
        setTimeout(function () {
            // Display greeting message after voting
            greetingMessage.style.display = 'block';
        }, 1000); // Delayed by 1 second for demonstration
    });
});
