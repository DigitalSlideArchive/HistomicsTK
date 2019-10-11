import { wrap } from 'girder/utilities/PluginUtils';
import { restRequest } from 'girder/rest';
import events from 'girder/events';
import ItemView from 'girder/views/body/ItemView';

import '../stylesheets/views/itemList.styl';

wrap(ItemView, 'render', function (render) {
    function quarantine(event) {
        const item = this.model;
        restRequest({
            type: 'PUT',
            url: 'HistomicsTK/quarantine/' + item.id,
            error: null
        }).done((resp) => {
            events.trigger('g:alert', {
                icon: 'ok',
                text: 'Item quarantined.',
                type: 'success',
                timeout: 4000
            });
            this.render();
        }).fail((resp) => {
            events.trigger('g:alert', {
                icon: 'cancel',
                text: 'Failed to quarantine item.',
                type: 'danger',
                timeout: 4000
            });
        });
    }

    this.once('g:rendered', function () {
        if (this.$el.find('.g-edit-item[role="menuitem"]').length && !this.$el.find('.g-histomicstk-quarantine-item[role="menuitem"]').length) {
            this.$el.find('.g-edit-item[role="menuitem"]').parent('li').after(
                '<li role="presentation"><a class="g-histomicstk-quarantine-item" role="menuitem"><span>Q</span>Quarantine item</a></li>'
            );
        }
        if (this.$el.find('.g-item-actions-menu').length && !this.$el.find('.g-histomicstk-open-item[role="menuitem"]').length &&
            this.model.attributes.largeImage) {
            this.$el.find('.g-item-actions-menu').prepend(
                `<li role="presentation">
                <a class="g-histomicstk-open-item" role="menuitem" href="/histomicstk#?image=${this.model.id}" target="_blank">
                    <i class="icon-link-ext"></i>Open in HistomicsTK
                </a>
            </li>`
            );
        }
        this.events['click .g-histomicstk-quarantine-item'] = quarantine;
        this.delegateEvents();
    });
    render.call(this);
});
