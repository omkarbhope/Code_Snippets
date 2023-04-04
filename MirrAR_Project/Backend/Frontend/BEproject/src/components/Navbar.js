import React, { useState } from 'react'
import logo from '../images/Logo.png'
import { Link } from "react-router-dom";


export default function Navbar() {

    const[bartext, setBartext] = useState("")
    const[show__afterclick, setShow__afterclick] = useState("")

    const toggleClass = () => {
        if(bartext == "toggle" && show__afterclick == "show__afterclick") {
            setBartext("")
            setShow__afterclick("")
        } else {
            setBartext("toggle")
            setShow__afterclick("show__afterclick")
        }
    }

  return (
    <>

    <header className = "header__nav">
        <nav className = "navbar__nav">
            <div className="logo">
                <Link to="/"><img src={logo}/></Link>
            </div>
            <div className="navlinks">
                <Link to="/tryout"><h3>Try-Out</h3></Link>
                {/* <Link to="/lipproducts"><h3>Products</h3></Link> */}
                <Link to="/tshirt"><h3>T-Shirts</h3></Link>
                <Link to="/necklace"><h3>Necklace</h3></Link>
            </div>
            <div className={`bars ${bartext}`} onClick={toggleClass}>
                <div className="line1"></div>
                <div className="line2"></div>
                <div className="line3"></div>
            </div>
        </nav>
    </header>
    <div className={`hidenshow__onclick ${show__afterclick}`}>

        <div className="navlinks__mobview">
            <button>Try-Out</button>
            <button>T-Shirts</button>
            <button>Necklace</button>
        </div>
            
    </div>

    
    
    
    </>
  )
}
