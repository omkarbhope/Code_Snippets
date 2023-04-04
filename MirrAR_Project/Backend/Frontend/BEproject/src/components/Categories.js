import React from 'react'
import { Link } from 'react-router-dom'

import lips from '../images/lips.png'
import essentials from '../images/essentials.png'
import eyes from '../images/eyes.png'
import face from '../images/face.png'




export default function Categories(props) {

  let categories = [lips, essentials, eyes, face]

  

  const getContent = () => {
    let content =[];

    for(let i = 0; i < 4; i++) {
      if (i==0) {
        content.push(
          <div className="category__bar">
            <Link to="/lipproducts"><img src={categories[i]} /></Link>
        </div>
        )
      }
      else if (i==1) {
        content.push(
          <div className="category__bar">
            <Link to="/tshirt"><img src={categories[i]} /></Link>
        </div>
        )
      }
      else {
        content.push(
          <div className="category__bar">
            <Link to="/tshirt"><img src={categories[i]} /></Link>
        </div>
        )
      }
    }

    return content;
  }



  return (
    <div className="categories__bars">      
        {getContent()}
    </div>
  )



  
}


