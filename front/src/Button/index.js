import React from 'react';

export default class Button extends React.Component {

    renderButton = ({ text, mod, onClick }) =>
        <button className={ `button${mod ? ' button-' + mod : ''}` } onClick = { onClick }>
            {text}
        </button>

    render = () => this.renderButton( this.props )

}