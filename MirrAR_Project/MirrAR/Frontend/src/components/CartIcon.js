import React from 'react'
import carticon from '../images/shopping-cart.png'
import { Link } from "react-router-dom";

export default function CartIcon(props) {
  return (
    <Link to="/checkout">
      <div class="carticon__image">
          <img src={carticon}/>
          {/* <span>{props.countCart == undefined ? 0 : props.countCart}</span> */}
      </div>
    </Link>
    
  )
}
