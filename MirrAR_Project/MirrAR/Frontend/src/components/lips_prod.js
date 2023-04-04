import React from 'react'
import { Link } from "react-router-dom";

import arrow from '../images/arrow_back_24px.png'
import bag from '../images/bag.png'
import addphoto from '../images/icons8-add-image-30.png'
import arrow2 from '../images/expand_more_24px.png'
import color1 from '../images/color1.png'
import dot from '../images/dot.png'
import lips1 from '../images/pac1.jpg'
import lips2 from '../images/pac2.jpg'

export default function lips_prod() {
  return (
    <>
      <div class="container-fluid cont-2 rounded bg-light p-3 my-3 border">
                <div className="row row1">
                    <div className="col-sm-1">
                            <img src={lips1} className="lips1"></img>
                    </div>
                    <div className="col-sm-11">
                            <h1 className="float-left">PAC Smudge Me Not Lipstick</h1> 
                    </div>
                </div>
                <div className="row">
                    <div className="col-sm-1">
                    </div>
                    <div className="col-sm-4 mt-2">
                        <img src={color1} className="smallcolor1"></img>
                        <input type="range" min="20" max="100" className="slider1 ml-4" id="myRange" onchange ="updatePigmentation()"></input>
                    </div>
                    <div className="col-sm">
                        <h1 className="float-right">₹450/-</h1>
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
    </>
  )
}
