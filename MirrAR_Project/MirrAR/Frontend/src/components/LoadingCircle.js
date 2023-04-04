import React from 'react'
import loading from '../images/ilpstick.gif'

export default function LoadingCircle() {
  return (
    <div>
        <div className="loading__screen__main">
            <img className = "loading__image" src={loading} alt = "loading image"/>
        </div>
    </div>
  )
}
