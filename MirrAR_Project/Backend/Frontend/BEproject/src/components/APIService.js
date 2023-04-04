import React, { Component } from 'react'

export class APIService extends Component {
    static getData(body) {
        fetch('http://127.0.0.1:8080/get_rgb', {
            methods : 'POST',
            headers : {
                'Content-Type' : 'application.json'
            },
            body : JSON.stringify(body)
    })
        .then(response => response.json())
        .catch(error => console.log(error))
    }
}

export default APIService