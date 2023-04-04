import React from 'react'
import { Link } from "react-router-dom";
import { useState, useEffect } from 'react';
import $ from 'jquery';
import ReactDOM from "react-dom";

import arrow from '../images/arrow_back_24px.png'
import bag from '../images/bag.png'
import addphoto from '../images/icons8-add-image-30.png'
import arrow2 from '../images/expand_more_24px.png'
import color1 from '../images/color1.png'
import dot from '../images/dot.png'
import lips1 from '../images/pac1.jpg'
import lips2 from '../images/pac2.jpg'

export default function Tryout() {
    const [red, setRed] = useState(0)
    const [blue, setBlue] = useState(0)
    const [green, setGreen] = useState(0)

    const updateColor = (r,b,g) => {
        setRed(r);
        setBlue(b);
        setGreen(g);
        const bodyStr = {r : red, g: green, b: blue};
        fetch('http://127.0.0.1:8080/post_rgb', {
            method : "POST",
            headers: {
                "Content-type": "application/json; charset=UTF-8"
            },
            mode: 'cors',
            body : JSON.stringify(bodyStr)
    })
        .then(response => response.json())
        .catch(error => console.log(error))

    }

    function storeIntoSession() {
        var imageProduct = sessionStorage.getItem("lip_img")
        var descProduct = sessionStorage.getItem("lip_desc")
        var priceProduct = sessionStorage.getItem("lip_price")

        var sessionCheckoutList = JSON. parse(sessionStorage.getItem("checkoutlist"))
        sessionCheckoutList.push(
            JSON.stringify({
                "imageProduct": imageProduct,
                "descProduct" : descProduct,
                "priceProduct" : priceProduct
            })
        )

        console.log(sessionStorage.setItem("checkoutlist",JSON.stringify(sessionCheckoutList)))
        alert("Item added to cart")

    }



  return (
    <>
        {/*<div className="ar-window">
            <video onloadedmetadata="onPlay(this)" id="inputVideo" autoplay></video>
            <canvas id="overlay"></canvas>
        </div>
        <div className="App">
            <a href="#">
                <img src={arrow} className="arrow"></img>
            </a>
        </div>*/}
        <div className="container-fluid cont-outline">
            <div class="ar-window">
                <video onloadedmetadata="onPlay(this)" id="inputVideo" autoplay></video>
                <canvas id="overlay"></canvas>
            </div>
            {/* <br></br>   */}
                <div className="row">
                    <div className="col-sm">
                        <div className="row-sm">
                            <Link to='/lipproducts'><img src={arrow} className="arrow float-left"></img></Link>
                        </div>
                    </div>
                    <div className="col-sm">
                        <div className="btn-grp float-right">
                            <button className="btn button ml-4" type="button">
                                Checkout <span className="caret"></span>
                            </button>
                            <button className="btn button ml-4" type="button" value="cart">
                                <Link to="/checkout"><img src={bag} className="bag_img"></img> <span className="caret"></span></Link>
                            </button>
                            <button className="btn button ml-4" type="button">
                                <img src={addphoto}></img> <span className="caret"></span>
                            </button>
                        </div>
                    </div>
                </div>
                <hr></hr>   
            <div className='container d-flex justify-content-center cont-tryon rounded'>
                <iframe
                        src={'http://127.0.0.1:8080/video_feed_makeup'}
                        width={650} 
                        height={450}
                        allow="autoplay; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen>
                </iframe>
            </div>
            <div className="row">
                <div className="col-sm">
                    <Link to="/">
                        <button className="btn button float-left" type="button">
                        All Products <span className="caret"></span>
                        </button>
                    </Link>
                </div>
                <div className="col-sm">
                    <button onClick={storeIntoSession} className="btn button float-right" type="button">
                    + Add Product <span className="caret"></span>
                    </button>
                </div>
            </div>
            <hr></hr>
            <div class="container-fluid cont-2 rounded p-3 mt-4 border">
                <div className="row row1">
                    <div className="col-sm-1">
                            <img src={sessionStorage.getItem("lip_img")} className="lips1"></img>
                    </div>
                    <div className="col-sm-11">
                            <h1 className="float-left">{sessionStorage.getItem("lip_desc")}</h1> 
                    </div>
                </div>
                <div className="row">
                    <div className="col-sm-1">
                    </div>
                    <div className="col-sm-4 mt-2">
                        <button className="color1 rounded-circle mr-3" onClick={(e) => updateColor(144,0,31)} id="color1"></button>
                        <button className="color2 rounded-circle mr-3" onClick={(e) => updateColor(219,7,52)} id="color1"></button>
                        <button className="color3 rounded-circle mr-3" onClick={(e) => updateColor(163,14,45)} id="color1"></button>
                        <button className="color4 rounded-circle mr-3" onClick={(e) => updateColor(114,22,41)} id="color1"></button>
                        <button className="color5 rounded-circle mr-3" onClick={(e) => updateColor(109,5,5)} id="color1"></button>
                        <button className="color6 rounded-circle mr-3" onClick={(e) => updateColor(114,22,41)} id="color1"></button>
                        <button className="color7 rounded-circle mr-3" onClick={(e) => updateColor(65,0,14)} id="color1"></button>
                        <button className="color8 rounded-circle mr-3" onClick={(e) => updateColor(114,22,41)} id="color1"></button>
                        {/* <input type="range" min="20" max="100" className="slider1 ml-4" id="myRange" onchange ="updatePigmentation()"></input> */}
                    </div>
                    <div className="col-sm">
                        <h1 className="float-right">{sessionStorage.getItem("lip_price")}</h1>
                    </div>
                    <div className="col-sm">
                        <button className="btn button float-right" type="button">
                            Add to Bag <span className="caret"></span>
                        </button>
                    </div>
                </div>
                {/*<div className="row">
                    <div className="col-sm-2">
                        <img src={color1} className="smallcolor1"></img>
                    </div>
                    <div className="col-sm-6">
                        <input type="range" min="20" max="100" value="60" className="" id="myRange" onchange ="updatePigmentation()"></input>
                    </div>
    </div>
                
                <div className="">
                    <input type="range" min="20" max="100" value="60" className="slider" id="myRange" onchange ="updatePigmentation()"></input>
                </div>
                <h1 className="price1">₹450/-</h1>
                <img className="dot1" src={dot}></img>
                <img className="dot2" src={dot}></img>
                <img className="dot3" src={dot}></img>
                <h1 className="othershades">Other Shades</h1>
                <button className="color1" id="color1"></button>
                <button className="color2" onClick="updateColor('90001F')"></button>
                <button className="color3" onClick="updateColor('DB0734')"></button>
                <button className="color4" onClick="updateColor('a30e2d')"></button>
                <button className="color5" onClick="updateColor('721629')"></button>
                <button className="color6" onClick="updateColor('6d0505')"></button>
                <button className="color7" onClick="updateColor('41000e')"></button>
                <button className="color8" onClick="updateColor('721629')"></button>
                {/*<div className="" dat-image="../images/pac1.jpg" data-name="PAC Smudge Me Not Lipstick"
                    data-price="450" data-id="0">
                    <img className="lips1" src={lips1} alt="" />
                    <input type="hidden" className="count" value="1" />
                    <button className="btn2">
                        <h1 className="addtobag">Add to Bag</h1>
                    </button>
    </div>*/}
            </div>
            <br/>
        </div>
        
            {/*<a href="#">
                <button className="btn button" type="button" style="width: 100%">
                Checkout <span className="caret"></span>
                </button>
            </a>

            {/*<div className="textbackground">
                <h1 className="text">Press anywhere on screen to click a photo</h1>
            </div>   

            <button className="addtocart" id="cartbutton" value="cart">
                <img src={bag} className="bag_img"></img>
            </button>

            <div className="addphoto">
                <img src={addphoto} className="images"></img>
            </div>

            <div className="medium-4  columns" id="cart">

                <div className="cart">
                    <div className="row">
                        <div className="medium-6 columns">
                            <button className="tiny secondary" id="clear">Clear the cart</button>
                        </div>

                    </div>
                    <div id="cartItems">Loading cart...</div>
                    Total price: <strong id="totalPrice">0 $</strong>
                </div>

            </div>

            <div className="addproduct">
                <h1>All Products </h1>
            </div>

            <img className="arrow2" src={arrow2}></img>

            <button className="addbutton" onclick="#">
                <h1 className="add">+ Add Product</h1>
            </button>

            <div className="lips">Lips</div>

            <hr className="lipsline"></hr>

            <div className="box1">
                <h1 className="name1">PAC Smudge Me Not Lipstick</h1>
                <img src={color1} className="smallcolor1"></img>
                <div className="slidecontainer">
                    <input type="range" min="20" max="100" value="60" className="slider" id="myRange" onchange ="updatePigmentation()"></input>
                </div>
                <h1 className="price1">₹450/-</h1>
                <img className="dot1" src={dot}></img>
                <img className="dot2" src={dot}></img>
                <img className="dot3" src={dot}></img>
                <h1 className="othershades">Other Shades</h1>
                <button className="color1" id="color1"></button>
                <button className="color2" onClick="updateColor('90001F')"></button>
                <button className="color3" onClick="updateColor('DB0734')"></button>
                <button className="color4" onClick="updateColor('a30e2d')"></button>
                <button className="color5" onClick="updateColor('721629')"></button>
                <button className="color6" onClick="updateColor('6d0505')"></button>
                <button className="color7" onClick="updateColor('41000e')"></button>
                <button className="color8" onClick="updateColor('721629')"></button>
                <div className="product medium-4 columns" dat-image="../images/pac1.jpg" data-name="PAC Smudge Me Not Lipstick"
                    data-price="450" data-id="0">
                    <img className="lips1" src={lips1} alt="" />
                    <input type="hidden" className="count" value="1" />
                    <button className="btn2">
                        <h1 className="addtobag">Add to Bag</h1>
                    </button>
                </div>
            </div>
        </div>

        < className="box2">
            <h1 className="name2">Soft Matte Cream Lipstick</h1>
            <button className="smallcolor3"></button>
            <h1 className="price2">₹635/-</h1>
            <img className="dot4" src={dot}></img>
            <img className="dot5" src={dot}></img>
            <img className="dot6" src={dot}></img>
            {/*<!-- <button className="btn3">
                    <h1 className="addtobag2">Add to Bag</h1>
  </button>-->
            <div className="product medium-4 columns" data-name="Soft Matte Cream Lipstick" data-price="653" data-id="1">
                <img className="lips2" src={lips2} alt="" />
                <input type="hidden" className="count" value="1" />
                <button className="btn3">
                    <h1 className="addtobag2">Add to Bag</h1>
                </button>
            </div>
        </div>











        {/*<div className="face">Face</div>*/}

        {/*<div> className="box3"></div>*/}
    </>
  )
}
