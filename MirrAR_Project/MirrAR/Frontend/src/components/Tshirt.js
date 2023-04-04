import React from 'react'
import tshirt_up from '../images/tshirt_up.jpg'
import { Link } from 'react-router-dom'

import lightblue from '../images/1.png'
import red from '../images/2.png'
import green from '../images/3.png'
import darkblue from '../images/4.png'
import turquoise from '../images/5.png'

const tshirt1 = (r) => {
    const x = r;
    console.log(x);
    
    sessionStorage.setItem("shirt_img",lightblue);
    sessionStorage.setItem("shirt_desc","LIGHT BLUE");
    sessionStorage.setItem("shirt_price","₹450");
}

const tshirt2 = (r) => {
    console.log(r);
    
    sessionStorage.setItem("shirt_img",red);
    sessionStorage.setItem("shirt_desc","PLAIN RED");
    sessionStorage.setItem("shirt_price","₹635");
}

const tshirt3 = (r) => {
    console.log(r);
    
    sessionStorage.setItem("shirt_img",green);
    sessionStorage.setItem("shirt_desc","PLAIN GREEN");
    sessionStorage.setItem("shirt_price","₹575");
}

const tshirt4 = (r) => {
    console.log(r);
    
    sessionStorage.setItem("shirt_img",darkblue);
    sessionStorage.setItem("shirt_desc","DARK BLUE");
    sessionStorage.setItem("shirt_price","₹725");
}

const tshirt5 = (r) => {
    console.log(r);
    
    sessionStorage.setItem("shirt_img",turquoise);
    sessionStorage.setItem("shirt_desc","TURQUOISE");
    sessionStorage.setItem("shirt_price","₹250");
}

export default function LipProducts() {
  return (
    <> 
    <div className="container-fluid body2">
        <div className="row">
            <div className="col-sm">
                <div className="row-sm">
                    <img src={tshirt_up} className="lipsticks img-fluid"></img>
                    <h1 className="carousel-caption">Choose Your Perfect Fit!</h1>
                </div>
            </div>
            
        </div>
    
    </div>
    
    <div className="container-fluid body2">
        <div className="row mt-5">
            <div className="col-sm-2 d-flex justify-content-center mb-4 ml-4 mr-4">
                <Link to={{pathname: "/tryout_shirt", state: {type: "1"}}} onClick={(e) => tshirt1('1')}>
                    <div className="pac_product1">
                        <img src={lightblue} className="pac_image1"></img>
                        <h1 className="pac1_desc">LIGHT BLUE <br/><br/>₹450</h1>
                    </div>
                </Link>
            </div>

            <div className="col-sm-2 d-flex justify-content-center mb-4 ml-4 mr-4">
            <Link to={{pathname: "/tryout_shirt", state: {type: "2"}}} onClick={(e) => tshirt2('2')}>
                <div className="pac_product1">
                    <img src={red} className="pac_image1"></img>
                    <h1 className="pac1_desc">PLAIN RED<br/><br/>₹635</h1>
                </div>
            </Link>
            </div>

            <div className="col-sm-2 d-flex justify-content-center mb-4 ml-4 mr-4">
            <Link to={{pathname: "/tryout_shirt", state: {type: "3"}}} onClick={(e) => tshirt3('3')}>
                <div className="pac_product1">
                    <img src={green} className="pac_image1"></img>
                    <h1 className="pac1_desc">PLAIN GREEN <br/><br/>₹575</h1>
                </div>
            </Link>
            </div>

            <div className="col-sm-2 d-flex justify-content-center mb-4 ml-4 mr-4">
            <Link to={{pathname: "/tryout_shirt", state: {type: "4"}}} onClick={(e) => tshirt4('4')}>
                <div className="pac_product1">
                    <img src={darkblue} className="pac_image1"></img>
                    <h1 className="pac1_desc">DARK BLUE<br/><br/>₹725</h1>
                </div>
            </Link>
            </div>

            <div className="col-sm-2 d-flex justify-content-center mb-4 ml-4 mr-4">
            <Link to={{pathname: "/tryout_shirt", state: {type: "5"}}} onClick={(e) => tshirt5('5')}>
                <div className="pac_product1">
                    <img src={turquoise} className="pac_image1"></img>
                    <h1 className="pac1_desc">TURQUOISE<br/><br/>₹250</h1>
                </div>
            </Link>
            </div>
        </div>
    </div>
       
    {/*
    <a href="#">
        <div className="pac_product1">
            <img src={lips1} className="pac_image1"></img>
            <h1 className="pac1_desc">MATTE LIPSTICK <br/><br/>₹450</h1>
        </div>
    </a>

    <a href="#">
        <div className="pac_product2">
            <img src={lips2} className="pac_image2"></img>
            <h1 className="pac2_desc">SOFT MATTE CREAM LIPSTICK <br/><br/>₹635</h1>
        </div>
    </a>

    <a href="#">
        <div className="pac_product3">
            <img src={lips3} className="pac_image3"></img>
            <h1 className="pac3_desc">LIP DIP LIPSTICK <br/><br/>₹575</h1>
        </div>
    </a>

    <a href="#">
        <div className="pac_product4">
            <img src={lips4} className="pac_image4"></img>
            <h1 className="pac4_desc">PURE MATTE LIPSTICK <br/><br/>₹725</h1>
        </div>
    </a>

    <a href="#">
        <div className="pac_product5">
            <img src={lips5} className="pac_image5"></img>
            <h1 className="pac5_desc">LIP COLOR REFILL<br/><br/>₹250</h1>
        </div>
  </a>*/}
</>
  )
}
